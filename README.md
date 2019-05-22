# My OpenCV project

This is a project built in Python using OpenCV for the University of Canterbury Computer Vision course.

_**Final Goal**_

The final goal of this project is for a picture of the speed climbing wall to be produced *possibly using a simple SLAM algorithm?* with a line representing the *approximate* center of gravity of the climber as they ascended the wall.

_**Why?**_

Climbing is now an **Olympic sport**, finally! With that comes a step up in the level of training for a few top level athletes. I hope to create a tool which makes it possible to visualise how their center of gravity changed as they climbed. This is because the lower the horizontal varience, the more efficient they climbed and therefore, the lower their theoretical maximum speed and minimum time.

_**What to do, what to do?**_

This is a checklist for what I need to do and how I plan to do it. As I complete these tasks I will ~~cross them out~~ so I feel good about myself or something.

1. Identify a box which *approximates* the center of mass of a person from behind
2. Use a centroid algorithm to confirm the center
3. Track this point no matter what
4. Track a line as the person climbs
5. Produce an image of the whole climb without the person in it
6. Recombine the traced line and map of the wall
