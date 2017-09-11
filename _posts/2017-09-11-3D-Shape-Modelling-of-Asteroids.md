---
layout: post
title: "3D Shape Modelling of Asteroids"
categories: journal
tags: [documentation,sample]
image:
  feature: NASA2.jpg
  teaser: NASA2-teaser.jpg
  credit:
  creditlink:
---

This article describes my summer at the [NASA Frontier Development Lab](http://www.frontierdevelopmentlab.org/#/).

### Machine Learning and Planetary Defence

As a researcher at NASA Frontier Development Lab (FDL), I spent my summer working alongside a team of experts with the aim of using machine learning to improve the process of modelling the 3D shapes of asteroids. FDL is an experimental tool in NASA’s innovation portfolio that emphasises artificial intelligence, inter-disciplinary approaches, rapid iteration and teamwork to produce breakthroughs useful to the space program.

Our team’s mission was to make use of all available radar observations to build 3D shape models of asteroids. The mission was motivated by a need to characterise as many Near-Earth Asteroids (NEAs) as possible by accurately estimating shapes, sizes, and physical properties. This is with the overarching aim of efficient detection and deflection of potential threats away from the Earth. Hence it is no surprise that our team operated under the arm of Planetary Defence, along with the Long-Period Comets team.

In more detail, our modelling challenge required optimising a plethora of parameters ranging from spin-axis orientation and period of rotation to the actual 3D shape. The current process of going from the observation data to a final model takes many months. Therefore we implemented various machine learning (ML) techniques with the aim of speeding up the process.

***

<center>
	<img src="{{ site.github.url }}/images/image_03_hw1-voxels.jpeg" alt="3d_model" style="width:20em;">
</center>
<sub>
<sub>
<em>	
	<center>
			<sub><strong>A radar-derived shape of the asteroid (8567) 1996 HW1 (Magri et al. 2011)</strong></sub>  
			<sub>represented as a triangular mesh (left) and as voxels (right).</sub>
	</center>
</em>
</sub>
</sub>

***

Our predominant form of input data consisted of a series of delay-Doppler images. These are radar images that are defined in both time delay and frequency. Although to the untrained eye (mine 8 weeks ago) these images might look like they are optical images, they actually have a non-unique mapping to the true asteroid shape. This many-to-one relationship added an extra level of complexity to the already difficult challenge of going from 2D to 3D representations. In order to go about solving this task we applied deep learning architectures such as autoencoders, variational autoencoders and generative adversarial networks to generate asteroid shapes and achieved promising results. We hope to continue this work over the next few months with the aim of publishing results for other researchers to build on in the future.

In addition to applying deep learning to produce a 3D shape model, another task was the determination of the asteroid spin state parameters. This task can take many days, so to speed up this process, we concentrated on using Bayesian optimisation to intelligently select highly probable parameter values. One of our key challenges was to reconfigure this ML technique to the spherical coordinate system that defines the domain of the asteroid spin axis. We found success porting Bayesian optimisation to a sphere and managed to reduce the number of calls to the expensive shape modelling software by an order of magnitude. 

***

<center>
	<img src="{{ site.github.url }}/images/image_01_artificial-shapes.jpeg" alt="training_data" style="width:30em;">
</center>
<sub>
<sub>
<em>	
	<center>
			<sub><strong>Synthetic asteroid shapes.</strong></sub>  
			<sub> A neural network needs a large
set of examples to learn what asteroids look like. There aren't enough
real asteroid shapes available for training, so the researchers at FDL
had to use randomly generated synthetic shapes (the software used to
obtain shapes was developed at the Poznań Astronomical Observatory).
After ensuring the shapes resemble real ones, they were combined with
different rotation states and plausible orbits to generate synthetic
radar observations.</sub>
	</center>
</em>
</sub>
</sub>

***

Our final addition to the asteroid shape modelling pipeline was the incorporation of a clustering algorithm to help scientists preprocess the large amount of delay-Doppler input data. This preprocessing algorithm resulted in reducing the task from one that takes a few days to one that takes a few hours. This was a clear example of how ML researchers can work together with domain-specific experts to speed up aspects of their day-to-day work. The application of ML tools enable the domain scientists to focus their efforts on more interesting areas of their field, without being held back by less meaningful tasks that can be readily automated. It highlights the healthy relationship that is possible when applying ML to new unexplored domains.

Overall I would like to thank all the FDL team for a great summer and the opportunity to apply ML in the exciting area of Planetary Defence. I now have more experience applying ML to a variety of tasks and I look forward to taking that with me on returning back to my PhD. Special mention goes to my team members Agata Rożek, Grace Young, Sean Marshall and our great mentors [Michael Busch](https://www.seti.org/users/michael-busch), [Chedy Raïssi](http://orpailleur.loria.fr/index.php/User:Chedy_Raïssi) and [Yarin Gal](http://mlg.eng.cam.ac.uk/yarin/). We further appreciate the fantastic hardware provided by IBM and NVIDIA.

<center>
	<img src="{{ site.github.url }}/images/NASA3.png" alt="Team_fdl" style="width:20em;">
</center>
<em>
		<sub>
			<center>
			<strong>The Asteroid Shape Modelling Team</strong>
			</center>
		</sub>
	<sub>
		Left: [Sean Marshall](http://astro.cornell.edu/~seanm/)\\
		Top: [Grace Young](http://www.graceunderthesea.com)\\
		Right: Me!\\
		Bottom: [Agata Rożek](http://astro.kent.ac.uk/~arozek1/)
	</sub>
</em>