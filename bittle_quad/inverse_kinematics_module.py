import numpy as np
import math

# Variables
d0 = 45		#Base to shoulder link length
a1 = 45		#Shoulder to elbow link length
d3 = 55		#Elbow to end effector link length

ROUNDING_ERROR = 10

def compute_angles(x, y, z):
	theta_base = round(math.atan2(y, (x-a1/2)), ROUNDING_ERROR)	

	sin_theta_elbow = round((((x-a1/2) * (x-a1/2) + (y-a1) * (y-a1) + (z - d0) * (z - d0)) - a1 * a1 - d3 * d3) / (2 * a1 * d3), ROUNDING_ERROR)

	if (sin_theta_elbow) <= 1 and (sin_theta_elbow) >=- 1:
		
		theta_elbow = round(math.asin(sin_theta_elbow), ROUNDING_ERROR)
		theta_elbow_possible = round(math.pi - theta_elbow, ROUNDING_ERROR)

		denominator = (x-a1/2) * (x-a1/2) + (y-a1) * (y-a1) + (z - d0) * (z - d0)

		if theta_base != 90:
			# Calculate (x-a1/2) / cos(theta_base)
			numerator_theta_shoulder = (z - d0) * (a1 + d3 * math.sin(theta_elbow)) \
										+ d3 * math.cos(theta_elbow) * (x-a1/2) / math.cos(theta_base)

			numerator_theta_shoulder_possible = (z - d0) * (a1 + d3 * math.sin(theta_elbow_possible)) + \
												d3 * math.cos(theta_elbow_possible) * (x-a1/2) / math.cos(theta_base)

			theta_shoulder = round(math.asin(numerator_theta_shoulder / denominator), ROUNDING_ERROR)
			theta_shoulder_possible = round(math.asin(numerator_theta_shoulder_possible / denominator), ROUNDING_ERROR)
			
		elif theta_base != 0:
			# Calculate (y-a1) / sin(theta_base)
			numerator_theta_shoulder = (z - d0) * (a1 + d3 * math.sin(theta_elbow)) \
										+ d3 * math.cos(theta_elbow) * (y-a1) / math.sin(theta_base)
			numerator_theta_shoulder_possible = (z - d0) * (a1 + d3 * math.sin(theta_elbow_possible)) \
												+ d3 * math.cos(theta_elbow_possible) * (y-a1) / math.sin(theta_base)
												
			theta_shoulder = round(math.asin(numerator_theta_shoulder / denominator), ROUNDING_ERROR)
			theta_shoulder_possible = round(math.asin(numerator_theta_shoulder_possible / denominator), ROUNDING_ERROR)

	else:

		print ("Point is not in range due to constraint in link length")
		return [[None, None, None],[None, None, None]]

	result = [[math.degrees(theta_base), math.degrees(theta_shoulder), math.degrees(theta_elbow)], \
			[math.degrees(theta_base), math.degrees(theta_shoulder_possible), math.degrees(theta_elbow_possible)]]

	# Return set of possible angles in degrees
	return result