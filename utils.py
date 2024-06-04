import math

def pi_to_pi(angle):
    # Normalize angle to be between -pi to pi
    return (angle + math.pi) % (2*math.pi) - math.pi