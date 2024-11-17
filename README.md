Academic Year: 2024/2025

* Assignment #4 Particle filter localization:
    - Objectives: Localize a forklift using the LiDAR and landmarks as reference
    - Tasks evaluated (15 points):	
        + Particle filter works and localizes the vehicle during the whole simulation (7 points) 
			* In the code you will find different TODOs. The main tasks are related to the initialization, prediction, update, resampling and etc... (see the slides and source code)
            * The random and guess initialization implementation is 1 point
            * The quality of the code and optimizations of the code will be positively evaluated (try to surprise me)
			* Note: Please send me the source code of the best localization solution you are able to achieve.

        + Report describing the particle filter performance under different scenarios. Just one-two-three paragraphs by scenario. Each paragraph should include a discussion of the trajectory estimated and execution time (3 points)
			* As we have seen during the lectures, the particle filter performance can be influenced by different factors (number of particles, error of the sensors or motion model...).
			* After executing the particle filter algorithm, a file called "res.txt" will be produced. This file will contain information regarding the estimation in X,Y coordinates of the best particle; the ground truth in X,Y coordinates; and the execution time of your solution (from initialization to resampling). All this information is used to compute the output that can be seen after executing "plotter.py"
			* The more the scenarios the higher the grade. Originality and quality of the report will be considered to determine grades (number of scenarios required is 3)
			* Note: The report shall include the description of the scenario, the trajectory of the forklift and the configuration used.
        
        + Implement your own resampling method and explain it (2 points)
            * https://bisite.usal.es/archivos/resampling_methods_for_particle_filtering_classification_implementation_and_strategies.pdf

	    + Implement any functionality on top of the particle filter. Optimize the code, any improvement will be positively (very) evaluated (3 points or more)
            * Any other data association technique
            * Explore the idea of combining a particle filter with a Kalman filter to improve the localization process
            * Implement an adaptive particle filter where the number of particles and resampling frequency adjusts dynamically based on the uncertainty in the system
            * ...........

* ROS bag:
    - Download the log file here: https://cloud.hipert.unimore.it/s/3y4DgMQDbTGK6WL

* OS requirements:
	- ROS (the installation command depends on the Linux distribution)SS
		Example: sudo apt install ros-melodic-desktop-full (this one applies to Ubuntu 18)        
    
* Instructions to compile the code:
        Open one terminal and run the following command: roscore
        To compile the code run this command in a different terminal (in the root folder): catkin_make 
	    The executable can be found in: ./devel/lib/particle/particle_node (run this command in a terminal to start the execution)
        As soon as you launch the executable you can start the simulation. To do so, open another terminal and write:
            rosbag play --clock --hz=10 out.bag    // this command allows to read all the data at 10hz (we do this for avoiding a lot of predictions without updating)
        The executable generates a file called 'res.txt'
	To generate the plot just write: "python3 plotter.py" (pay attention to the output file called res.txt with the trajectory and execution time of your implementation) 
	
* Important Note:
    - All the TODOs are in the file called "particle_filter.cpp" & "main.cpp" but you are more than free to modify any source file
    - In the folder you find 2 txt files:
        * pf_slam.txt: Hipert's 'best' particle filter implementation
        * res.txt: Nacho's 'prototype' particle filter implementation (feel free to use my result as reference)

