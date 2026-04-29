(define (domain allactionrobot)
  (:requirements :strips :typing :negative-preconditions :fluents) 
  (:types robot object)
  (:predicates
    (at ?robot - robot ?object - object)
    (inaction ?robot - robot) 
    (holding ?robot - robot ?object - object)
    (at-location  ?object - object ?location - object)
    (toggled-on ?robot - robot ?object - object)
    (toggled-off ?robot - robot ?object - object)
    (object-open ?robot - robot ?object - object)
    (object-close ?robot - robot ?object - object)
    (break ?robot - robot ?object - object)
    (sliced ?object - object)
    (cleaned ?robot - robot ?object - object)
    (is-fridge ?object - object) 
  )
  (:functions
    (fridge-state ?fridge - object) ;; Numeric fluent for fridge state
  )

  (:action GoToObject
    :parameters (?robot - robot ?object - object)
    :precondition (not (inaction ?robot))

    :effect (and 
              (at ?robot ?object)
              (forall (?another_object - object)
                (when (at ?robot ?another_object)
                  (not (at ?robot ?another_object))
                )
              )
              (not (inaction ?robot))
            )
  )


  (:action PickupObject      
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and 
                    (at-location ?object ?location)
                    (at ?robot ?location)
                    (not(inaction ?robot))
    )
    :effect (and
              (holding ?robot ?object)
              (not(inaction ?robot))
    )
  )


  (:action PutObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and 
                    (holding ?robot ?object)
                    (not(inaction ?robot))
                    (at ?robot ?location)
                    (or (not (is-fridge ?location))  ;; Check if location is not the fridge
                        (> (fridge-state ?location) 0))  ;; If it is the fridge, it must be open
    )
    :effect (and
              (at-location ?object ?location)
              (not (holding ?robot ?object))
              (not(inaction ?robot))
    )
  )


  (:action ToggleObjectOn
    :parameters (?robot - robot ?object - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?object)
    )
    :effect (and
              (not(inaction ?robot))
              (toggled-on ?robot ?object)
    )
  )


  (:action ToggleObjectOff
    :parameters (?robot - robot ?object - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?object)
    )
    :effect (and
                (not(inaction ?robot))
                (toggled-off ?robot ?object)
    )
  )


  (:action OpenObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?object)
    )
      
    :effect (and
                (not(inaction ?robot))
                (object-open ?robot ?object)
    )
  )


  (:action BreakObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?object)
    )
    :effect (and
              (not(inaction ?robot))
              (break ?robot ?object)
    )
  )
 

  (:action CloseObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?object)
    )
    :effect (and
              (not(inaction ?robot))
              (object-close ?robot ?object)
  )
  )



  (:action SliceObject
    :parameters (?robot - robot ?object - object ?location - object)
    :precondition (and 
                    (at-location ?object ?location)
                    (at ?robot ?location)
                    (not(inaction ?robot))
    )
    :effect (and
              (not(inaction ?robot))
              (sliced ?object)
    )
  )    
  

 (:action CleanObject
    :parameters (?robot - robot ?object - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?object)
    )
    :effect (and
              (not(inaction ?robot))
              (cleaned ?robot ?object)
    )    
  )
  (:action OpenFridge
    :parameters (?robot - robot ?fridge - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?fridge)
                    (is-fridge ?fridge)  ;; Ensure the object is a fridge
    )
    :effect (and
                (not(inaction ?robot))
                (object-open ?robot ?fridge)
                (increase (fridge-state ?fridge) 1)  
    )
  )

  (:action CloseFridge
    :parameters (?robot - robot ?fridge - object)
    :precondition (and
                    (not(inaction ?robot))
                    (at ?robot ?fridge)
                    (object-open ?robot ?fridge) 
                    (is-fridge ?fridge)  
    )
    :effect (and
              (not(inaction ?robot))
              (object-close ?robot ?fridge)
              (decrease (fridge-state ?fridge) 1)  
    )
  )
)

