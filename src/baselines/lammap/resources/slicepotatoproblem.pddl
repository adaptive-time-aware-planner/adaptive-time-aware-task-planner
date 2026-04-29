(define (problem slice-potato-problem)
  (:domain allactiondomain)

  (:objects
    robot1 - robot
    kitchen - location
    potato1 - object
    knife1 - object   ; Assuming knife is needed and treated as an object
  )

  (:init
    (robot-at robot1 kitchen)          ; Robot is initially in the kitchen
    (object-at potato1 kitchen)        ; Potato is also in the kitchen
    (object-at knife1 kitchen)         ; Knife is also in the kitchen
    (whole potato1)                    ; Potato is whole and needs to be sliced
    (not (holding robot1 knife1))      ; Robot is not initially holding the knife
  )

  (:goal
    (and
      (sliced potato1)                 ; The goal is for the potato to be sliced
    )
  )
)