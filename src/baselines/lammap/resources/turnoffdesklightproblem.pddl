(define (problem turn-off-desk-light)
  (:domain allactiondomain)

  (:objects
    robot1 - robot
    desk - location
    desk_light - object
  )

  (:init
    (robot-at robot1 desk)          ; Robot is initially at the desk
    (object-at desk_light desk)     ; Desk light is at the desk
    (object-on desk_light)          ; Desk light is initially on
  )

  (:goal
    (not (object-on desk_light))    ; The goal is for the desk light to be off
  )
)