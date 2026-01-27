##High-level Task Specification##

We have a constellation of low-earth orbit satellites. Each satellite can be described using a two-line element set or TLE. The goal is to calculate and visualize the coverage results. That is, given a point on the earth's surface and a minimum look angle, we can determine whether a satellite is in view of that point at a given moment in time. With this calculation performed at points distributed over the Earth and then calculated out over a long period of time (90 days), then we can get an estimate of the percent of time that a given position on the Earth has satellite coverage.

###Task 1 - Single-Satellite, Single-Point, Single Time Instance###

- Import an SGP4 propagator
- Propagate forward 1 satellite position to a single time and verify the position versus known test vectors
- Calculate the look angle at a single time / position on the Earth's surface and verify the angle versus known test vectors

###Task 2 - Algorithm enhancement for satellite propagation###

- Define an intelligent algorithm for propagating time forward and repeating the calculation. What this means is that if the satellite is on the other side of the Earth, there is no need to check the next second for the visibility calculation.

###Task 3 - Implement orbit propagation through the entire time horizon###

- Given the algorithm defined in Task 2, propagate the satellite position forward in time and calculate the look angle at each instant.

###Task 4 - Algorithm enhancement to extend through a large number of points###

- If the satellite is at a negative number, there is no reason to calculate the look angle for a neighbor point. Instead, the calculation can start from a coarse grained look over the entire earth and then zoom in with a finer grain whereever the edges of the look angles are.

###Task 5 - Extend the calculations through a number of satellites###

- Increase the number of satellites beyond 1 and calculate through the time horizon. Visibility of 1 satellite is sufficient at a given point in time.

###Task 6 - Visualize###

- Generate a simple reactive map interface that can be usable through the browser.
