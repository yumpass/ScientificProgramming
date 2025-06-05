Specklegrams are created by a process called modal interference, which happens at the end of a multimode optical fiber. The dimensions of a multimode fiber (MMF) allow several light modes at different wavelengths to propagate, meaning that electromagnetic waves travel along several specific paths.

When a laser beam goes into an optical fiber, it creates different ways the light can travel. This depends on the size of the fiber. There's a relationship between the diameter of the core and the cladding, and the number of ways the light can travel. Speckle or specklegrams are created from these optical paths and the phase delays of each mode. This pattern is a changing spatial distribution of intensities, where constructive interference creates areas of maximum intensity, while total destructive interference between modes produces areas of no light.

![image](https://github.com/user-attachments/assets/bf25ee97-88f1-4a20-bce4-84bbba1ecbbd)


The specklegram has great metrological utility, as the patterns generated can give information about the disturbances made along the fiber. The prediction of the magnitude of these disturbances will depend on the mathematical tools used.

Synthetic specklegram simulations were performed using the Finite Element Method (FEM) within COMSOL Multiphysics, integrated with Matlab. This model simulated the propagation of an optical field through a multimode optical fiber, concentrating specifically on the sensing region affected by temperature fluctuations. This method facilitated an accurate approximation of the behavior of the fiber under perturbations, excluding undisturbed regions to optimize computational efficiency [1].
Specklegrams are created by a process called modal interference, which happens at the end of a multimode optical fiber. The dimensions of a multimode fiber (MMF) allow several light modes at different wavelengths to propagate, meaning that electromagnetic waves travel along several specific paths.

When a laser beam goes into an optical fiber, it creates different ways the light can travel. This depends on the size of the fiber. There's a relationship between the diameter of the core and the cladding, and the number of ways the light can travel. Speckle or specklegrams are created from these optical paths and the phase delays of each mode. This pattern is a changing spatial distribution of intensities, where constructive interference creates areas of maximum intensity, while total destructive interference between modes produces areas of no light.

![image](https://github.com/user-attachments/assets/e72bb953-6bf9-46e4-99ab-4a08ea18f2bc)


The specklegram has great metrological utility, as the patterns generated can give information about the disturbances made along the fiber. The prediction of the magnitude of these disturbances will depend on the mathematical tools used.

Synthetic specklegram simulations were performed using the Finite Element Method (FEM) within COMSOL Multiphysics, integrated with Matlab. This model simulated the propagation of an optical field through a multimode optical fiber, concentrating specifically on the sensing region affected by temperature fluctuations. This method facilitated an accurate approximation of the behavior of the fiber under perturbations, excluding undisturbed regions to optimize computational efficiency [1].

Through this FEM model, the vector wave equation 1 was numerically solved for each propagation mode within the multimode optical fiber (MMF) under analysis [1, 2].

$$
\nabla\times\nabla\times\vec{E}-k_0^2n^2\vec{E}=0 \enspace
$$

![image](https://github.com/user-attachments/assets/ad6f1da5-14a4-4f48-a4cd-a61d82100019)


Here $\vec{E}$, represents the electric field of each mode, $k_0$ is the wavenumber in vacuum, and n stands for the refractive index of the MMF. The refractive index can further be updated in response to thermal fluctuations using Equation 2.

$$
n\approx n_0+C_{T0}(T-T_0)\enspace
$$

Where $C_{TO}$ is the thermo-optic coefficient, $n_0$ the reference index, $T_0$  the reference temperature, and T the temperature to be measured.The initial core refractive index is calculated using the Sellmeier equation [1], while the cladding refractive index $n_{0cla}$ is given by equation 3.

$$
n_{0cla} = \sqrt{n^2_{0co} - NA^2} \enspace
$$

Where $n_0co$ is the initial core refractive index and NA is the numerical aperture.

This are the optical parametters of the dataset we show as an example:

<table style="width:20%">
<tr>
<th>Numerical aperture (nm)</th>
<th>0.13 </th>
</tr>

<tr>
<td>Wavelength (nm)</td>
<td>632.8 </td>
</tr>

<tr>
<td>Core Diameter (µm) </td>
<td>9</td>
</tr>

<tr>
<td>Cladding Diameter (µm)       </td>
<td>40</td>
</tr>

<tr>
<td>Core Index  </td>
<td>1.457 </td>
</tr>

<tr>
<td>Cladding Index   </td>
<td>1.4521 </td>
</tr>

<tr>
<td>length of the Perturbation (mm)</td>
<td>0.3 </td>
</tr

<tr>
<td>Temperature Range (°C) </td>
<td>0 to 100</td>
</tr>

<tr>
<td>Step (°C)</td>
<td>0.1</td>
</tr>

<tr>
<td>Number of images </td>
<td>1001 </td>
</tr>
</table>

 [1] Juan David Arango Moreno, Yeraldin Velez, Victor Aristizabal, Francisco Velez, Gómez Alberto, Jairo Quijano, and Jorge
 Herrera Ramirez. Numerical study using finite element method for the thermal response of fiber specklegram sensors with
 changes in the length of the sensing zone. Computer Optics, 45:534–540, 07 2021.
 
 [2] Luis Castaño, Luis Gutierrez, Jairo Quijano, Jorge Herrera-Ramírez, Alejandro Hoyos, Francisco Vélez, Víctor Aristizabal,
 Luiz Silva-Nunez, and Jorge Gómez. Temperature measurement by means of fiber specklegram sensors (fss). Óptica Pura y
 Aplicada, 2018.
