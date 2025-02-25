# A-Star Demo
A program that uses A-star search to find a path between the pink dot (start) and the green dot (goal). By default the A-star search is weighted using `WEIGHT = 1.2`, but you can play around with this setting by changing it in the code.

![demo](assets/demo.gif)

# Installation of Dependencies
Run the following command
```bash
pip install -r requirements.txt
```

# Running the Program
```bash
python3 .
```

# Controls (keys)
+ `←, →, ↑, ↓`: move the pink dot around the world (arrow keys).
+ `r`: generate a new map.
+ `a`: set mode to _animation_ (searching is done in small increments, each one drawn on screen).
+ `s`: set mode to _static_ (searching is completed and the result is drawn on screen).