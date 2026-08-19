#!/usr/bin/env bash
# Current five-task prompt-v3 evaluation table. Sourced by the policy launcher.

TASKS=(
  "Pick up the black pen, perform a right-to-left handover, and insert the pen into the black pen holder on the left side."
  "Put the pill box into the center of the notebook."
  "Perform a 90-degree clockwise rotation of the screwdriver using the right arm."
  "Pull the pill box onto the black pad beside it."
  "Press the green button once to switch the indicator light from green to red."
)

load_prompt_variants() {
  case "$1" in
    0)
      PROMPT_VARIANTS=(
        "Pick up the black pen, perform a right-to-left handover, and insert the pen into the black pen holder on the left side."
        "Pick up the black pen with the right arm, pass it to the left arm, and place it into the black holder on the left."
        "Use the right gripper to grasp the black pen, hand it over to the left gripper, then insert it into the black pen holder."
        "Take the black pen using the right hand. Transfer it to the left hand and put it in the left-side black pen holder."
        "Perform a right-to-left handover of the black pen and insert the pen into the black holder on the left."
        "Grasp the black pen on the right, give it to the left arm, and seat it in the black pen holder."
        "Move the black pen from the right arm to the left arm, then place it in the black holder to the left."
        "With the right arm, pick up the black pen; pass it to the left arm and put it into the left black pen holder."
        "Hand the black pen from right to left, then insert it into the black pen holder on the left."
        "Pick the black pen, transfer it across to the left gripper, and deposit it in the left-side black holder."
      )
      ;;
    1)
      PROMPT_VARIANTS=(
        "Put the pill box into the center of the notebook."
        "Move the pill box to the center of the notebook."
        "Place the pill box in the middle of the notebook."
        "Pick up the pill box and set it down at the notebook's center."
        "Relocate the pill box onto the center area of the notebook."
        "Put the pill container in the center of the notebook."
        "Grasp the pill box and position it in the middle of the notebook."
        "Transfer the pill box to the notebook and place it centrally."
        "Set the pill box down on the notebook's central region."
        "Take the pill box and place it at the center of the notebook."
      )
      ;;
    2)
      PROMPT_VARIANTS=(
        "Perform a 90-degree clockwise rotation of the screwdriver using the right arm."
        "Use the right arm to turn the screwdriver 90 degrees clockwise."
        "Rotate the screwdriver clockwise by a quarter turn with the right arm."
        "With the right hand, make a 90-degree clockwise rotation of the screwdriver."
        "Turn the screwdriver one quarter turn clockwise using the right arm."
        "Use the right gripper to rotate the screwdriver 90 degrees in the clockwise direction."
        "Give the screwdriver a clockwise quarter-turn with the right arm."
        "Rotate the screwdriver to the right by 90 degrees."
        "Using the right arm, twist the screwdriver clockwise through 90 degrees."
        "Perform a right-arm clockwise quarter-turn of the screwdriver."
      )
      ;;
    3)
      PROMPT_VARIANTS=(
        "Pull the pill box onto the black pad beside it."
        "Pull the pill box onto the adjacent black pad."
        "Drag the pill box onto the black pad beside it."
        "Move the pill box by pulling it onto the nearby black pad."
        "Pull the pill container onto the black pad next to it."
        "Bring the pill box onto the neighboring black pad."
        "Use a pulling motion to place the pill box onto the black pad."
        "Draw the pill box over to the black pad beside it."
        "Pull the pill box from its current position onto the adjacent black pad."
        "Slide the pill box onto the black pad next to it."
      )
      ;;
    4)
      PROMPT_VARIANTS=(
        "Press the green button once to switch the indicator light from green to red."
        "Press the green button once so that the indicator changes from green to red."
        "Push the green button one time to switch the light from green to red."
        "Turn the indicator red by pressing the green button once."
        "Press the green control once and change the indicator light from green to red."
        "Make the light change from green to red with a single press of the green button."
        "Tap the green button once to set the indicator to red."
        "Use one press on the green button to switch the indicator light to red."
        "Single-press the green button until the green indicator becomes red."
        "Press the green button a single time, causing the light to turn red."
      )
      ;;
    *)
      echo "TASK_ID must be one of 0,1,2,3,4; got '$1'" >&2
      exit 2
      ;;
  esac
}
