import math
import random
import matplotlib.pyplot as plt


# METRICS: scalar field + probability evaluation

#PHI----------------------------------------------------------------
def d_phi(phi, mu_phi):
    return math.atan2(math.sin(phi - mu_phi), math.cos(phi - mu_phi))
#AMPLITUDE------------------------------------------------------------
def d_amp(a, mu_a):
    return ((a - mu_a) + (60*min(0,(a - mu_a)**3)))
#DELTA----------------------------------------------------------------
def Delta(phi, amp):
    mu_phi = math.pi
    mu_amp = 0.5

    sigma_phi = 0.6
    sigma_amp = 0.2

    dp = d_phi(phi, mu_phi)
    da = d_amp(amp, mu_amp)
    dist2 = - (dp*dp)/(1*sigma_phi*sigma_phi) - (da*da)/(1/3*sigma_amp*sigma_amp)
    return dist2
#probability-----------------------------------------------------------------------------
def probability(phi, amp):
    return math.exp(Delta(phi, amp))


# GENERATION: produces raw structured samples ----------------------------------------------------

#Generates raw (phase, amplitude) sample.
def generate_sample():
    phi = random.uniform(0, 2 * math.pi)
    amp = random.uniform(0, 1)
    return phi, amp

#Rejection sampling using probability field.
def generate(n):
    
    points = []
    max_p = 1.0  # exp(Phi) ≤ 1

    while len(points) < n:
        phi, amp = generate_sample()
        P = probability(phi, amp)
        if random.uniform(0, max_p) < P:
            points.append({
                "phase": phi,
                "amplitude": amp,
                "probability": P
            })

    return points

# =========================================================
# Example usage
# =========================================================

if __name__ == "__main__":
    data = generate(5000)
    #print(data[:5])
    
    phsae = [d["phase"]*180/math.pi for d in data]
    amplitude = [d["amplitude"]for d in data]
    plt.scatter(phsae,amplitude, s= 1)
    plt.xlim(0,360)
    plt.ylim(0,1)
    plt.plot
    plt.show()
