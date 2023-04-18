import numpy as np
import matplotlib.pyplot as plt

# Signal length
signal_length = 1000

# Generate random complex signal
random_signal = np.random.randn(signal_length) + 1j * np.random.randn(signal_length)

# Gram-Schmidt orthogonalization
orthogonalized_signal = np.zeros_like(random_signal, dtype=complex)
for i in range(signal_length):
    orthogonalized_signal[i] = random_signal[i]
    for j in range(i):
        orthogonalized_signal[i] -= np.vdot(orthogonalized_signal[j], random_signal[i]) / np.linalg.norm(orthogonalized_signal[j]) ** 2 * orthogonalized_signal[j]

# Generate BPSK signal
bpsk_data = np.random.randint(0, 2, signal_length) * 2 - 1 # Generate random binary data
bpsk_signal = bpsk_data.astype(complex) # Convert binary data to complex symbols (-1, 1)

# Gram-Schmidt orthogonalization
orthogonalized_bpsk_signal = np.zeros_like(bpsk_signal, dtype=complex)
for i in range(signal_length):
    orthogonalized_bpsk_signal[i] = bpsk_signal[i]
    for j in range(i):
        orthogonalized_bpsk_signal[i] -= np.vdot(orthogonalized_bpsk_signal[j], bpsk_signal[i]) / np.linalg.norm(orthogonalized_bpsk_signal[j]) ** 2 * orthogonalized_bpsk_signal[j]

# Generate QPSK signal
qpsk_data = np.random.randint(0, 4, signal_length)
constellation = np.array([-1 - 1j, -1 + 1j, 1 - 1j, 1 + 1j]) / np.sqrt(2) # QPSK constellation
qpsk_signal = constellation[qpsk_data]

# Gram-Schmidt orthogonalization
orthogonalized_qpsk_signal = np.zeros_like(qpsk_signal, dtype=complex)
for i in range(signal_length):
    orthogonalized_qpsk_signal[i] = qpsk_signal[i]
    for j in range(i):
        orthogonalized_qpsk_signal[i] -= np.vdot(orthogonalized_qpsk_signal[j], qpsk_signal[i]) / np.linalg.norm(orthogonalized_qpsk_signal[j]) ** 2 * orthogonalized_qpsk_signal[j]

# Generate QAM signal
M = 16  # QAM modulation order
qam_data = np.random.randint(0, M, signal_length)
constellation = np.array([-3 - 3j, -3 - 1j, -3 + 1j, -3 + 3j,
                          -1 - 3j, -1 - 1j, -1 + 1j, -1 + 3j,
                          1 - 3j, 1 - 1j, 1 + 1j, 1 + 3j,
                          3 - 3j, 3 - 1j, 3 + 1j, 3 + 3j]) / np.sqrt(10) # QAM constellation
qam_signal = constellation[qam_data]


# Gram-Schmidt orthogonalization
orthogonalized_qam_signal = np.zeros_like(qam_signal, dtype=complex)
for i in range(signal_length):
    orthogonalized_qam_signal[i] = qam_signal[i]
    for j in range(i):
        orthogonalized_qam_signal[i] -= np.vdot(orthogonalized_qam_signal[j], qam_signal[i]) / np.linalg.norm(orthogonalized_qam_signal[j]) ** 2 * orthogonalized_qam_signal[j]


plt.figure(figsize=(12, 10))

# Plot constellation diagram of original and orthogonalized signals
plt.subplot(221)
plt.scatter(np.real(random_signal), np.imag(random_signal), label='Original Signal')
plt.scatter(np.real(orthogonalized_signal), np.imag(orthogonalized_signal), label='Orthogonalized Signal')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Random Signal Constellation Diagram')
plt.legend()
plt.grid(True)

# Plot constellation diagram of original and orthogonalized signals
plt.subplot(222)
plt.scatter(np.real(bpsk_signal), np.imag(bpsk_signal), label='Original Signal')
plt.scatter(np.real(orthogonalized_bpsk_signal), np.imag(orthogonalized_bpsk_signal), label='Orthogonalized Signal')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('BPSK Constellation Diagram')
plt.legend()
plt.grid(True)

# Plot constellation diagram of original and orthogonalized signals
plt.subplot(223)
plt.scatter(np.real(qpsk_signal), np.imag(qpsk_signal), label='Original Signal')
plt.scatter(np.real(orthogonalized_qpsk_signal), np.imag(orthogonalized_qpsk_signal), label='Orthogonalized Signal')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('QPSK Constellation Diagram')
plt.legend()
plt.grid(True)


# Plot constellation diagram of original and orthogonalized signals
plt.subplot(224)
plt.scatter(np.real(qam_signal), np.imag(qam_signal), label='Original Signal')
plt.scatter(np.real(orthogonalized_qam_signal), np.imag(orthogonalized_qam_signal), label='Orthogonalized Signal')
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('QAM Constellation Diagram')
plt.legend()
plt.grid(True)

plt.show()