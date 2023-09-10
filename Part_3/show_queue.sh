#!/bin/bash

# Clear the screen initially
# clear

while true; do
  # Run your command here (replace with your actual command)
  # For example, running the 'date' command
  squeue -u s4501559

  # Sleep for 10 seconds
  sleep 5

  # Clear the previous two lines (move up two lines, clear, and move down)
  tput cuu1    # Move cursor up one line
  tput cuu1    # Move cursor up one more line
  #tput cuu1
  #tput cuu1
  tput ed      # Clear to the end of the screen
#  tput cud1    # Move cursor down one line
done
