(define (problem wash-tomato-problem)
  (:domain allactiondomain)
  
  (:objects
    robot1 - robot
    kitchen - location
    sink - location
    tomato1 - object
  )

  (:init
    (robot-at robot1 kitchen)          ; Robot is initially at the kitchen
    (object-at tomato1 kitchen)        ; Tomato is initially at the kitchen
    (path-exists kitchen sink)         ; There's a path from the kitchen to the sink
    (reachable robot1 sink)            ; Robot can reach the sink
    (dirty tomato1)                    ; The tomato is initially dirty
  )

  (:goal
    (and
      (clean tomato1)                  ; The goal is for the tomato to be clean
    )
  )
)