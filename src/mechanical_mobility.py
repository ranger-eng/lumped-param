import numpy
import matplotlib.pyplot as plt

class MechanicalMobility:
    real_pt = None
    imag_pt = None

    def __init__(self, omega):
        self.omega = numpy.asarray(omega)
        self.freq = omega / (2 * numpy.pi)

    def mobility(self):
        if self.real_pt is None or self.imag_pt is None:
            raise ValueError("real_pt and imag_pt must be set before calculating mobility.")
        return self.real_pt + 1j * self.imag_pt

    def magnitude(self):
        mag = numpy.sqrt(self.real_pt**2 + self.imag_pt**2)
        return mag

    def phase(self):
        phase = numpy.arctan2(self.imag_pt, self.real_pt)
        return phase

    def spring(self, Cm):
        self.real_pt = numpy.zeros(len(self.omega))
        self.imag_pt = Cm * self.omega

    def mass(self, Mm):
        self.real_pt = numpy.zeros(len(self.omega))
        self.imag_pt = -1 / (Mm * self.omega)

    def damper(self, Rm):
        self.real_pt = Rm
        self.imag_pt = numpy.zeros(len(self.omega))

class MechanicalOperations:
    @staticmethod
    def series(mobility1: MechanicalMobility, mobility2: MechanicalMobility):
        combined = MechanicalMobility(mobility1.omega)

        # Mobilities in series add
        combined.real_pt = mobility1.real_pt + mobility2.real_pt
        combined.imag_pt = mobility1.imag_pt + mobility2.imag_pt
        return combined

    @staticmethod
    def parallel(mobility1: MechanicalMobility, mobility2: MechanicalMobility):
        combined = MechanicalMobility(mobility1.omega)

        complex_mobility1 = mobility1.mobility()
        complex_mobility2 = mobility2.mobility()

        combined_mobility = 1 / (1 / complex_mobility1 + 1 / complex_mobility2)

        combined.real_pt = numpy.real(combined_mobility)
        combined.imag_pt = numpy.imag(combined_mobility)

        return combined

    @staticmethod
    def bode_plot(*args, labels=None, freq_axis='freq'):
        # Create single figure for all plots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        if freq_axis == 'omega':
            freq_values = args[0].omega
            x_label = 'Angular Frequency (rad/s)'
        else:
            freq_values = args[0].freq
            x_label = 'Frequency (Hz)'

        for i, mobility in enumerate(args):
            label = labels[i] if labels and i < len(labels) else f'Mobility {i+1}'

            # Plot on the same axes
            ax1.plot(freq_values, 20*numpy.log10(mobility.magnitude()), label=label)
            ax2.plot(freq_values, mobility.phase() * 180/numpy.pi, label=label)

        # Configure axes
        ax1.set_title("Bode Plot - Magnitude")
        ax1.set_ylabel("Magnitude (dB)")
        ax1.grid(True)
        ax1.set_xscale('log')
        ax1.legend()

        ax2.set_title("Bode Plot - Phase")
        ax2.set_xlabel(x_label)
        ax2.set_ylabel("Phase (degrees)")
        ax2.grid(True)
        ax2.set_xscale('log')
        ax2.legend()

        plt.tight_layout()