# -*- coding: utf-8 -*-
"""LMP(Language Model-based Programs) 생성을 위한 유틸리티 모듈.

이 모듈은 언어 모델을 활용하여 동적으로 코드를 생성하고 실행하는
LMP, LMPFGen, LMP_wrapper 클래스와 관련 헬퍼 함수들을 포함합니다.
AI2-THOR와 같은 시뮬레이션 환경에서 자연어 명령을 코드로 변환하고
실행하는 데 사용됩니다.
"""
import ast
import json
import os
import re
import sys
import traceback
from time import sleep
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import astunparse
from dotenv import load_dotenv
from openai import OpenAI
from pygments import highlight
from pygments.formatters import TerminalFormatter
from pygments.lexers import PythonLexer

from src.utils.common import create_module_logger

# LMP(Language Model Program)를 위한 임포트


base_dir = os.path.dirname(os.path.abspath(__file__))
load_dotenv()

logger = create_module_logger(__name__, module_log=True)
openai_api_key = os.environ.get("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OPENAI_API_KEY 환경 변수를 찾을 수 없습니다.")
    raise EnvironmentError("OPENAI_API_KEY 환경 변수를 찾을 수 없습니다.")

client = OpenAI(api_key=openai_api_key)

# 계획 로그를 저장할 파일 열기
log_file_path = os.path.join(base_dir, "../result/cap_plan.txt")
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
log_file = open(log_file_path, "w", buffering=1)


class LMP:
    """언어 모델 기반 프로그램(LMP)을 나타내는 클래스.

    LLM을 호출하여 주어진 쿼리로부터 코드를 생성하고,
    생성된 코드를 실행하며, 세션을 유지 관리하는 역할을 합니다.
    """

    def __init__(
        self,
        name: str,
        cfg: Dict[str, Any],
        lmp_fgen: "LMPFGen",
        fixed_vars: Dict[str, Any],
        variable_vars: Dict[str, Any],
    ):
        """LMP 인스턴스를 초기화합니다.

        Args:
            name (str): LMP의 이름.
            cfg (Dict[str, Any]): LMP 설정을 담은 딕셔너리.
            lmp_fgen (LMPFGen): 새 함수를 생성하는 데 사용될 함수 생성기.
            fixed_vars (Dict[str, Any]): 실행 중에 변경되지 않는 고정 변수.
            variable_vars (Dict[str, Any]): 실행 중에 변경될 수 있는 가변 변수.
        """
        self._name = name
        self._cfg = cfg
        self._base_prompt = self._cfg["prompt_text"]
        self._stop_tokens = list(self._cfg["stop"])
        self._lmp_fgen = lmp_fgen
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self.exec_hist = ""  # 실행 기록을 저장하는 변수
        self._max_retries = 3  # 자동 오류 수정을 위한 최대 재시도 횟수

    def clear_exec_hist(self) -> None:
        """실행 기록을 초기화합니다."""
        self.exec_hist = ""

    def build_prompt(
        self, query: str, context: str = "", **kwargs: Any
    ) -> Tuple[str, str]:
        """LLM에 전달할 프롬프트를 구성합니다.

        기본 프롬프트, 변수 목록, 세션 기록, 컨텍스트, 쿼리를 조합하여
        최종 프롬프트를 생성합니다.

        Args:
            query (str): 사용자의 주된 요청 또는 질문.
            context (str, optional): 쿼리에 대한 추가적인 문맥. Defaults to "".
            **kwargs: 프롬프트 문자열 포맷팅에 사용될 키워드 인자.

        Returns:
            Tuple[str, str]: 생성된 전체 프롬프트와 사용된 쿼리 문자열.
        """
        if len(self._variable_vars) > 0:
            # 사용 가능한 함수들을 import 문 형식으로 변환합니다.
            variable_vars_imports_str = (
                f"from utils import {', '.join(self._variable_vars.keys())}"
            )
        else:
            variable_vars_imports_str = ""

        # 기본 프롬프트 포맷팅
        prompt = self._base_prompt.format(
            variable_vars_imports=variable_vars_imports_str, **kwargs
        )

        if self._cfg["maintain_session"]:
            # 세션 유지 설정이 켜져 있으면, 이전 실행 기록을 프롬프트에 추가
            prompt += f"\n{self.exec_hist}"

        if context != "":
            # 컨텍스트가 있으면 프롬프트에 추가
            prompt += f"\n{context.format(**kwargs)}"

        # 쿼리 접두사와 접미사를 붙여 최종 쿼리 문자열 생성
        use_query = f'{self._cfg["query_prefix"]}{query}{self._cfg["query_suffix"]}'
        prompt += f"\n{use_query}"

        return prompt, use_query

    def __call__(self, query: str, context: str = "", **kwargs: Any) -> Optional[Any]:
        """LMP를 함수처럼 호출하여 코드를 생성하고 실행합니다.

        1. `build_prompt`를 호출하여 프롬프트를 생성합니다.
        2. OpenAI API를 호출하여 코드를 생성합니다.
        3. 생성된 코드에서 새로운 함수가 필요한 경우 `lmp_fgen`으로 생성합니다.
        4. `exec_safe`를 사용하여 안전하게 코드를 실행합니다.
        5. 실행 중 오류가 발생하면, LLM에게 오류 수정을 요청하고 재시도합니다.
        6. 실행 결과를 반환하거나 세션을 업데이트합니다.

        Args:
            query (str): 사용자의 주된 요청 또는 질문.
            context (str, optional): 쿼리에 대한 추가적인 문맥. Defaults to "".
            kwargs: 코드 실행 시 지역 변수로 사용될 키워드 인자.

        Returns:
            Optional[Any]: 설정에서 반환 값이 지정된 경우, 실행 결과.
        """
        fix_it_prompt = None
        prompt, use_query = self.build_prompt(query, context=context, **kwargs)
        sys_guide = "You are a highly intelligent and context-aware Household AI Robot Assistant."
        guide = """
You must honor what is written in the note unconditionally.
## [Notes]
1. The # is a directive, and anything that follows it is code. When you see a directive, generate the Python code for it.
2. DO NOT include triple quotes (```) or the word 'python' in your response.
3. Never send me an empty python block.
4. Never include 'import'.

## [EXAMPLES]
"""
        prompt_msgs = [
            {"role": "system", "content": sys_guide},
            {"role": "user", "content": f"{guide}{prompt}"},
        ]

        # 자동 오류 수정을 위한 재시도 루프
        for attempt in range(self._max_retries):
            # OpenAI API 호출
            if attempt == 0:  # 최초 시도
                api_prompt_msgs = prompt_msgs
            else:  # 오류 수정 재시도
                # 이전 단계에서 생성된 오류 수정 프롬프트를 사용
                api_prompt_msgs = [{"role": "user", "content": fix_it_prompt}]

            while True:
                try:
                    completion = client.chat.completions.create(
                        messages=api_prompt_msgs,
                        temperature=self._cfg["temperature"],
                        model="gpt-4o",
                        max_tokens=self._cfg["max_tokens"],
                    )
                    code_str = completion.choices[0].message.content.strip()
                    code_str = code_str.replace("python", "").replace("```", "")
                    if not code_str:
                        raise ValueError("LLM returned an empty code block.")
                    break
                except Exception as e:
                    logger.error(f"OpenAI API 호출 중 오류 발생: {e}")
                    logger.info("10초 후 재시도합니다.")
                    sleep(10)

            # 컨텍스트와 실행할 코드를 조합합니다.
            if self._cfg["include_context"] and context != "":
                to_exec = f"{context}\n{code_str}"
                to_log = f"{context}\n{use_query}\n{code_str}"
            else:
                to_exec = code_str
                to_log = f"{use_query}\n{to_exec}"

            # 실행 내용을 로그 파일에 기록하고 콘솔에 출력합니다.
            log_file.write(to_log + "\n\n")

            if sys.stdout.isatty():
                # 터미널일 경우에만 색상 적용
                to_log_pretty = highlight(to_log, PythonLexer(), TerminalFormatter())
            else:
                # 파일 로그일 경우 색상 없이 출력
                to_log_pretty = to_log

            logger.info(
                f"LMP {self._name} PROMPT (Attempt {attempt + 1}):\n\n{api_prompt_msgs}\n"
            )
            if attempt > 0:
                print(
                    f"LMP {self._name} trying to fix code (Attempt {attempt + 1}):\n\n{to_log_pretty}\n"
                )
            else:
                print(f"LMP {self._name} exec:\n\n{to_log_pretty}\n")

            # 생성된 코드에서 새로운 함수가 있는지 파싱하고 생성합니다.
            new_fs = self._lmp_fgen.create_new_fs_from_code(code_str)
            self._variable_vars.update(new_fs)

            # 실행에 사용할 전역 및 지역 변수를 설정합니다.
            gvars = merge_dicts([self._fixed_vars, self._variable_vars])
            lvars = kwargs

            if not self._cfg["debug_mode"]:
                try:
                    # 디버그 모드가 아닐 경우, 코드를 안전하게 실행합니다.
                    exec_safe(to_exec, gvars, lvars)
                    # 성공 시 루프 탈출
                    break
                except Exception as e:  # ActionFailedError 대신 모든 예외를 잡도록 수정
                    error_msg = traceback.format_exc()
                    logger.error(
                        "코드 실행 중 오류 발생 (Attempt %d):\n%s",
                        attempt + 1,
                        error_msg,
                    )

                    if attempt < self._max_retries - 1:
                        # 실패한 코드와 오류 메시지를 포함하여 LLM에게 수정을 요청하는 새로운 프롬프트를 구성합니다.
                        error_context = traceback.format_exc().splitlines()[-1]

                        # 디버깅 가이드라인 추가
                        debugging_guideline = (
                            "The previous code failed with the following error. "
                            "Please analyze the error message and the failed code to fix it.\n"
                            "--- FAILED CODE --- \n"
                            f"{code_str}\n"
                            "--- ERROR MESSAGE ---\n"
                            f"{error_context}\n"
                            "--- IMPORTANT DEBUGGING TIPS ---\n"
                            "1. If the error is 'Hand is already full', you MUST add a `put()` or `drop()` action before the failed `pickup()` action.\n"
                            "2. Double-check that the object IDs passed to functions are correct and available in the current context.\n"
                            "3. Ensure the sequence of actions is logical (e.g., you must `open()` a drawer before you can `put()` something in it).\n"
                            "--- INSTRUCTIONS ---\n"
                            "1. ONLY return the corrected Python code block.\n"
                            "2. Do NOT include any explanations, comments, or markdown formatting like ```python ... ```.\n"
                            "--- CORRECTED CODE ---\n"
                        )

                        fix_it_prompt = debugging_guideline

                        # 래퍼의 컨텍스트를 사용하여 수정된 코드를 생성합니다.
                        # 참고: self.wrapper.context는 이전 상호작용 기록을 포함하고 있습니다.
                        # 실패한 시도에 대한 컨텍스트를 명시적으로 추가하여 LLM이 실수를 반복하지 않도록 합니다.
                        log_file.write(
                            f"ATTEMPT {attempt + 1} FAILED. Trying to fix...\n\n"
                        )
                    else:
                        logger.error(
                            "최대 재시도 횟수(%d)에 도달했습니다. 자동 수정을 중단합니다.",
                            self._max_retries,
                        )
                        # 마지막 시도도 실패하면 예외를 다시 발생시켜 프로그램을 중단시킴
                        raise

        # 실행 기록을 업데이트합니다.
        self.exec_hist += f"\n{to_exec}"

        if self._cfg["maintain_session"]:
            # 세션 유지 시, 지역 변수를 가변 변수에 업데이트합니다.
            self._variable_vars.update(lvars)

        if self._cfg["has_return"]:
            # 반환 값이 필요한 경우, 지정된 변수를 반환합니다.
            return lvars[self._cfg["return_val_name"]]
        return None

    def extract_code_block(self, response: str) -> str:
        """LLM의 응답에서 순수한 Python 코드 블록만 추출합니다.

        Args:
            response (str): LLM이 반환한 전체 문자열 응답.

        Returns:
            str: 추출된 코드 블록 또는 추출 실패 시 원본 문자열의 일부.
        """
        # ```python ... ``` 블록을 찾는 정규 표현식
        code_match = re.search(r"```python\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # ``` ... ``` 블록을 찾는 정규 표현식
        code_match = re.search(r"```\n(.*?)\n```", response, re.DOTALL)
        if code_match:
            return code_match.group(1).strip()

        # 코드 블록 마커가 없는 경우, 응답에 코드와 설명이 섞여있을 수 있습니다.
        # "CORRECTED CODE" 이후의 내용을 코드로 간주합니다.
        if "--- CORRECTED CODE ---" in response:
            return response.split("--- CORRECTED CODE ---")[-1].strip()

        # 그래도 찾을 수 없으면, 잠재적인 설명 부분을 제거하려고 시도합니다.
        # 예를 들어, "Here's the corrected code:" 와 같은 문장.
        lines = response.split("\n")
        code_lines = []
        in_code = False
        for line in lines:
            # 코드처럼 보이는 첫 줄에서부터 추출 시작
            if not in_code and (
                line.strip().startswith("def")
                or line.strip().startswith("import")
                or line.strip().startswith("#")
            ):
                in_code = True
            if in_code:
                code_lines.append(line)

        if code_lines:
            return "\n".join(code_lines).strip()

        return response.strip()  # 최후의 수단으로 원본 반환

    def _get_query_str(self, query: str) -> str:
        """LMP 쿼리 문자열을 생성합니다."""
        return f"{self.query_prefix}{query}{self.query_suffix}"


class LMPFGen:
    """LMP를 위한 함수 생성기(Function Generator).

    LLM을 사용하여 주어진 함수 시그니처나 코드 조각으로부터
    동적으로 파이썬 함수를 생성합니다.
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        fixed_vars: Dict[str, Any],
        variable_vars: Dict[str, Any],
    ):
        """LMPFGen 인스턴스를 초기화합니다.

        Args:
            cfg (Dict[str, Any]): 함수 생성기 설정을 담은 딕셔너리.
            fixed_vars (Dict[str, Any]): 실행 중에 변경되지 않는 고정 변수.
            variable_vars (Dict[str, Any]): 실행 중에 변경될 수 있는 가변 변수.
        """
        self._cfg = cfg
        self._stop_tokens = list(self._cfg["stop"])
        self._fixed_vars = fixed_vars
        self._variable_vars = variable_vars
        self._base_prompt = self._cfg["prompt_text"]

    def create_f_from_sig(
        self,
        f_name: str,
        f_sig: str,
        other_vars: Optional[Dict[str, Any]] = None,
        fix_bugs: bool = False,
        return_src: bool = False,
    ) -> Union[Callable, Tuple[Callable, str]]:
        """함수 시그니처로부터 함수를 생성합니다.

        Args:
            f_name (str): 생성할 함수의 이름.
            f_sig (str): 생성할 함수의 시그니처 (예: "add(a, b)").
            other_vars (Optional[Dict[str, Any]], optional): 추가적인 변수. Defaults to None.
            fix_bugs (bool, optional): 생성된 코드의 버그를 수정할지 여부. Defaults to False.
            return_src (bool, optional): 생성된 함수와 함께 소스 코드를 반환할지 여부. Defaults to False.

        Returns:
            Union[Callable, Tuple[Callable, str]]: 생성된 함수 객체. `return_src`가 True이면
                                                  (함수 객체, 소스 코드) 튜플을 반환합니다.
        """
        print(f"Creating function: {f_sig}")

        use_query = f'{self._cfg["query_prefix"]}{f_sig}{self._cfg["query_suffix"]}'
        prompt = f"{self._base_prompt}\n{use_query}"
        sys_guide = "You are a highly intelligent and context-aware Household AI Robot Assistant."
        guide = """## [TASK]
1. The # is a directive, and anything that follows it is code. When you see a directive, generate the Python code for it.
2. Do not include triple quotes (```) or the word 'python' in your response.
3. Do not send blank

## [EXAMPLES]
"""
        prompt_msgs = [
            {"role": "system", "content": sys_guide},
            {"role": "user", "content": f"{guide}{prompt}"},
        ]

        while True:
            try:
                # LLM을 호출하여 함수 소스 코드 생성
                completion = client.chat.completions.create(
                    messages=prompt_msgs,
                    stop=self._stop_tokens,
                    temperature=self._cfg["temperature"],
                    model="gpt-4o",
                    max_tokens=self._cfg["max_tokens"],
                )
                f_src = completion.choices[0].message.content.strip()
                f_src = f_src.replace("python", "").replace("```", "")
                break
            except Exception as e:
                logger.error(f"OpenAI API 호출 중 함수 생성 오류: {e}")
                logger.info("10초 후 재시도합니다.")
                sleep(10)

        if fix_bugs:
            # 버그 수정 프롬프트를 통해 코드 개선
            fix_prompt_str = (
                f"# {f_src}\nFix the bug if there is one. Improve readability. "
                "Keep same inputs and outputs. Only small changes. No comments."
            )
            fix_prompt_msgs = [{"role": "user", "content": fix_prompt_str}]
            completion = client.chat.completions.create(
                model="gpt-4o", temperature=0, messages=fix_prompt_msgs
            )
            f_src = completion.choices[0].message.content.strip()
            f_src = f_src.replace("python", "").replace("```", "")

        if other_vars is None:
            other_vars = {}
        gvars = merge_dicts([self._fixed_vars, self._variable_vars, other_vars])
        lvars: Dict[str, Any] = {}

        # 생성된 소스 코드를 실행하여 함수 객체 획득
        exec_safe(f_src, gvars, lvars)
        f = lvars[f_name]

        # 생성된 함수와 소스 코드를 로깅
        to_print_str = f"{use_query}\n{f_src}"
        if sys.stdout.isatty():
            # 터미널일 경우에만 색상 적용
            to_print = highlight(to_print_str, PythonLexer(), TerminalFormatter())
        else:
            to_print = to_print_str

        log_file.write(f"{use_query}\n{f_src}\n\n")
        logger.info(f"LMP FGEN PROMPT:\n\n{prompt_msgs}\n")
        print(f"LMP FGEN created:\n\n{to_print}\n")

        if return_src:
            return f, f_src
        return f

    def create_new_fs_from_code(
        self,
        code_str: str,
        other_vars: Optional[Dict[str, Any]] = None,
        fix_bugs: bool = False,
        return_src: bool = False,
    ) -> Union[Dict[str, Callable], Tuple[Dict[str, Callable], Dict[str, str]]]:
        """주어진 코드에서 새로운 함수들을 찾아 생성합니다.

        코드를 파싱하여 이전에 정의되지 않은 함수 호출을 찾고,
        `create_f_from_sig`를 호출하여 해당 함수들을 동적으로 생성합니다.

        Args:
            code_str (str): 함수를 찾을 소스 코드 문자열.
            other_vars (Optional[Dict[str, Any]], optional): 추가적인 변수. Defaults to None.
            fix_bugs (bool, optional): 생성된 코드의 버그를 수정할지 여부. Defaults to False.
            return_src (bool, optional): 생성된 함수와 함께 소스 코드를 반환할지 여부. Defaults to False.

        Returns:
            Union[Dict[str, Callable], Tuple[Dict[str, Callable], Dict[str, str]]]:
            새로 생성된 함수들의 딕셔너리. `return_src`가 True이면
            (함수 딕셔너리, 소스 코드 딕셔너리) 튜플을 반환합니다.
        """
        fs: Dict[str, str] = {}
        f_assigns: Dict[str, str] = {}
        f_parser = FunctionParser(fs, f_assigns)
        f_parser.visit(ast.parse(code_str))
        for f_name, f_assign in f_assigns.items():
            if f_name in fs:
                fs[f_name] = f_assign

        if other_vars is None:
            other_vars = {}

        new_fs: Dict[str, Callable] = {}
        srcs: Dict[str, str] = {}
        for f_name, f_sig in fs.items():
            all_vars = merge_dicts(
                [self._fixed_vars, self._variable_vars, new_fs, other_vars]
            )
            # 아직 정의되지 않은 함수인 경우에만 생성
            if not var_exists(f_name, all_vars):
                f, f_src = self.create_f_from_sig(
                    f_name, f_sig, new_fs, fix_bugs=fix_bugs, return_src=True
                )

                # 생성된 함수의 본문에서 재귀적으로 자식 함수들을 생성
                f_def_body = astunparse.unparse(ast.parse(f_src).body[0].body)
                child_fs, child_f_srcs = self.create_new_fs_from_code(
                    f_def_body, other_vars=all_vars, fix_bugs=fix_bugs, return_src=True
                )

                if len(child_fs) > 0:
                    new_fs.update(child_fs)
                    srcs.update(child_f_srcs)

                    # 자식 함수가 생성되었으므로 부모 함수를 다시 정의하여 스코프에 포함
                    gvars = merge_dicts(
                        [self._fixed_vars, self._variable_vars, new_fs, other_vars]
                    )
                    lvars: Dict[str, Any] = {}
                    exec_safe(f_src, gvars, lvars)
                    f = lvars[f_name]

                new_fs[f_name], srcs[f_name] = f, f_src

        if return_src:
            return new_fs, srcs
        return new_fs


class FunctionParser(ast.NodeTransformer):
    """AST를 순회하며 함수 호출과 할당을 파싱하는 클래스."""

    def __init__(self, fs: Dict[str, str], f_assigns: Dict[str, str]):
        """FunctionParser 인스턴스를 초기화합니다.

        Args:
            fs (Dict[str, str]): 함수 시그니처를 저장할 딕셔너리.
            f_assigns (Dict[str, str]): 함수 할당문을 저장할 딕셔너리.
        """
        super().__init__()
        self._fs = fs
        self._f_assigns = f_assigns

    def visit_Call(self, node: ast.Call) -> ast.AST:
        """함수 호출 노드를 방문하여 시그니처를 추출합니다.

        Args:
            node (ast.Call): 방문한 함수 호출 노드.

        Returns:
            ast.AST: 처리된 노드.
        """
        self.generic_visit(node)
        if isinstance(node.func, ast.Name):
            f_sig = astunparse.unparse(node).strip()
            f_name = astunparse.unparse(node.func).strip()
            self._fs[f_name] = f_sig
        return node

    def visit_Assign(self, node: ast.Assign) -> ast.AST:
        """할당 노드를 방문하여 함수 할당을 추출합니다.

        Args:
            node (ast.Assign): 방문한 할당 노드.

        Returns:
            ast.AST: 처리된 노드.
        """
        self.generic_visit(node)
        if isinstance(node.value, ast.Call):
            assign_str = astunparse.unparse(node).strip()
            if isinstance(node.value.func, ast.Name):
                f_name = astunparse.unparse(node.value.func).strip()
                self._f_assigns[f_name] = assign_str
        return node


def var_exists(name: str, all_vars: Dict[str, Any]) -> bool:
    """주어진 변수 이름이 스코프 내에 존재하는지 확인합니다.

    Args:
        name (str): 확인할 변수의 이름.
        all_vars (Dict[str, Any]): 변수 스코프.

    Returns:
        bool: 변수가 존재하면 True, 그렇지 않으면 False.
    """
    try:
        eval(name, all_vars)
        return True
    except NameError:
        return False


def merge_dicts(dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """여러 딕셔너리를 하나로 병합합니다.

    Args:
        dicts (List[Dict[str, Any]]): 병합할 딕셔너리 리스트.

    Returns:
        Dict[str, Any]: 병합된 딕셔너리.
    """
    return {k: v for d in dicts for k, v in d.items()}


def exec_safe(
    code_str: str,
    gvars: Optional[Dict[str, Any]] = None,
    lvars: Optional[Dict[str, Any]] = None,
) -> None:
    """안전하게 코드 문자열을 실행합니다.

    'import', '__'와 같은 잠재적으로 위험한 키워드를 금지하고,
    'exec' 및 'eval' 내장 함수를 비활성화하여 코드를 실행합니다.

    Args:
        code_str (str): 실행할 코드 문자열.
        gvars (Optional[Dict[str, Any]], optional): 전역 변수. Defaults to None.
        lvars (Optional[Dict[str, Any]], optional): 지역 변수. Defaults to None.
    """
    banned_phrases = ["import", "__"]
    for phrase in banned_phrases:
        if phrase in code_str:
            raise ValueError(f"금지된 구문 '{phrase}'이 코드에 포함되어 있습니다.")

    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}

    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([gvars, {"exec": empty_fn, "eval": empty_fn}])

    exec(code_str, custom_gvars, lvars)


class LMP_wrapper:
    """AI2-THOR 환경을 위한 래퍼 클래스.

    LMP가 시뮬레이션 환경과 쉽게 상호작용할 수 있도록
    객체 정보 조회, 상태 확인 등의 API를 제공합니다.
    """

    # ROS 모드에서 사용하는 JSON 파일 경로
    _ROS_DYNAMIC_STATES_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../assets/runtime_state/dynamic/object_states.json",
    )
    _ROS_STATIC_INIT_STATES_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../assets/runtime_state/static/object_init_states.json",
    )
    _ROS_STATIC_ABILITIES_PATH = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "../../../../assets/runtime_state/static/object_abilities.json",
    )

    def __init__(self, controller: Any, cfg: Dict[str, Any], render: bool = False):
        """LMP_wrapper 인스턴스를 초기화합니다.

        Args:
            controller (Any): AI2-THOR 컨트롤러 인스턴스. ROS 모드에서는 None.
            cfg (Dict[str, Any]): 관련 설정 딕셔너리.
            render (bool, optional): 렌더링 여부. Defaults to False.
        """
        self.controller = controller
        self._cfg = cfg
        self.ros_mode = controller is None
        self.render = render

        if self.ros_mode:
            with open(self._ROS_STATIC_INIT_STATES_PATH, "r") as f:
                init_states = json.load(f)
            self.object_names = list(init_states.keys())
        else:
            self.object_names = list(
                set(
                    obj["objectType"].lower()
                    for obj in controller.step("Pass").metadata["objects"]
                )
            )

    def _load_dynamic_states(self) -> Dict[str, Any]:
        """ROS 모드에서 동적 객체 상태 JSON 파일을 읽어 반환합니다.

        Returns:
            Dict[str, Any]: 객체 이름을 키로, 상태 딕셔너리를 값으로 하는 딕셔너리.
        """
        with open(self._ROS_DYNAMIC_STATES_PATH, "r") as f:
            return json.load(f)

    def is_obj_visible(self, obj_name: str) -> bool:
        """특정 객체가 현재 시야에 보이는지 확인합니다.

        Args:
            obj_name (str): 확인할 객체의 이름.

        Returns:
            bool: 객체가 보이면 True, 그렇지 않으면 False.
        """
        if self.ros_mode:
            return True

        data = self.controller.step("Pass").metadata["objects"]
        visible_obj_names = [
            obj["objectType"].lower() for obj in data if obj["visible"]
        ]
        return obj_name in visible_obj_names

    def get_obj_names(self) -> List[str]:
        """현재 씬에 있는 모든 객체의 이름 목록을 반환합니다.

        Returns:
            List[str]: 객체 이름 목록.
        """
        return self.object_names[::]

    def get_obj_id(self, obj_name: str) -> Optional[str]:
        """객체 이름으로 객체 ID를 찾습니다. (정확한 일치)

        완전 관찰 가능 환경을 가정하고 모든 객체를 탐색합니다. 'Sink'를 찾을 경우
        씬에서 'SinkBasin'을 직접 찾아 그 ID를 반환하는 특수 처리를 포함합니다.

        Args:
            obj_name (str): ID를 찾을 객체의 이름.

        Returns:
            Optional[str]: 찾은 객체의 ID. 없으면 None을 반환합니다.
        """
        if not obj_name:
            return None

        obj_name_lower = obj_name.lower()

        # Special handling for 'sink' -> find a 'sinkbasin'
        if obj_name_lower == "sink":
            logger.info("Query is for 'Sink', redirecting to find 'SinkBasin'.")
            if self.ros_mode:
                # ROS 모드에서는 'sink|sinkbasin' 키를 사용
                return "sink|sinkbasin"
            for obj in self.controller.last_event.metadata["objects"]:
                if obj["objectType"] == "SinkBasin":
                    logger.info(f"Found SinkBasin with ID '{obj['objectId']}'.")
                    return obj["objectId"]
            logger.warning("Could not find any SinkBasin in the scene.")
            return None

        if self.ros_mode:
            return obj_name_lower

        # Normal object search (exact match)
        all_objects = self.controller.last_event.metadata["objects"]
        for obj in all_objects:
            if obj["objectType"].lower() == obj_name_lower:
                return obj["objectId"]

        return None

    def get_true_states(self, obj_id: str) -> List[str]:
        """특정 객체의 현재 상태 목록을 반환합니다.

        (예: isToggled, isCooked 등)

        Args:
            obj_id (str): 상태를 조회할 객체의 ID.

        Returns:
            List[str]: 객체의 현재 '참'인 상태 목록.
        """
        if self.ros_mode:
            dynamic_states = self._load_dynamic_states()
            obj_data = dynamic_states.get(obj_id, {})
            true_state = [
                key
                for key, value in obj_data.items()
                if isinstance(value, bool) and value is True
            ]
            print(f"{obj_id}의 현재 상태: {true_state}")
            return true_state

        properties = [
            "isInteractable",
            "isToggled",
            "isBroken",
            "isFilledWithLiquid",
            "isDirty",
            "isUsedUp",
            "isCooked",
            "isHeatSource",
            "isColdSource",
            "isSliced",
            "isOpen",
            "isPickedUp",
            "isMoving",
        ]
        true_state = []
        data = self.controller.step("Pass").metadata["objects"]
        for obj in data:
            if obj["objectId"] == obj_id:
                for prop in properties:
                    if obj.get(prop, False):
                        true_state.append(prop)
                break
        print(f"{obj_id}의 현재 상태: {true_state}")
        return true_state

    def get_ability_states(self, obj_id: str) -> List[str]:
        """특정 객체가 가질 수 있는 능력 상태 목록을 반환합니다.

        (예: toggleable, cookable 등)

        Args:
            obj_id (str): 능력을 조회할 객체의 ID.

        Returns:
            List[str]: 객체가 가진 능력 목록.
        """
        if self.ros_mode:
            if os.path.exists(self._ROS_STATIC_ABILITIES_PATH):
                with open(self._ROS_STATIC_ABILITIES_PATH, "r") as f:
                    abilities_data = json.load(f)
                return abilities_data.get(obj_id, [])
            return []

        properties = [
            "toggleable",
            "breakable",
            "canFillWithLiquid",
            "dirtyable",
            "canBeUsedUp",
            "cookable",
            "sliceable",
            "openable",
            "pickupable",
        ]
        ability_state = []
        data = self.controller.step("Pass").metadata["objects"]
        for obj in data:
            if obj["objectId"] == obj_id:
                for prop in properties:
                    if obj.get(prop, False):
                        ability_state.append(prop)
                break
        return ability_state

    def get_parentReceptacles(self, obj_id: str) -> Optional[List[str]]:
        """특정 객체를 담고 있는 부모 객체(receptacle) 목록을 반환합니다.

        Args:
            obj_id (str): 부모를 찾을 객체의 ID.

        Returns:
            Optional[List[str]]: 부모 객체의 ID 목록. 없으면 None을 반환합니다.
        """
        if self.ros_mode:
            dynamic_states = self._load_dynamic_states()
            obj_data = dynamic_states.get(obj_id, {})
            parentReceptacles = obj_data.get("parentReceptacles", None)
            print(f"{obj_id}의 부모 리셉터클: {parentReceptacles}")
            return parentReceptacles

        data = self.controller.step("Pass").metadata["objects"]
        parentReceptacles = None
        for obj in data:
            if obj["objectId"] == obj_id:
                parentReceptacles = obj["parentReceptacles"]
                break
        print(f"{obj_id}의 부모 리셉터클: {parentReceptacles}")
        return parentReceptacles

    def get_obj_in_hand(self) -> Optional[str]:
        """에이전트가 손에 들고 있는 객체의 ID를 반환합니다.

        Returns:
            Optional[str]: 손에 들고 있는 객체의 ID. 없으면 None을 반환합니다.
        """
        if self.ros_mode:
            dynamic_states = self._load_dynamic_states()
            for obj_name, obj_data in dynamic_states.items():
                receptacles = obj_data.get("parentReceptacles", None)
                if receptacles and "agent" in receptacles:
                    return obj_name
            return None

        data = self.controller.last_event.metadata["objects"]
        for obj in data:
            if obj.get("isPickedUp", False):
                return obj["objectId"]
        return None
