import numpy as np
import timeit
import scipy.stats
import scipy
from copy import deepcopy
from scipy.stats import norm
from scipy.stats import rayleigh
from scipy.optimize import curve_fit
from matplotlib import pyplot as plt
import tikzplotlib
import statsmodels.api as sm

## Main part with example application of code is at the end of the file

class Fading:
    def __init__(self, number_values=1000000, variance_gauss = 5, variance_rayleigh = 2.5, corr_gauss = 0.7, corr_rayleigh = 0.7, decorrelation=500, dec_factor = 10, p_save=""):
        #general
        self.l = number_values
        #gauss
        self.N_gauss = 100 # summands
        self.var_gauss = variance_gauss #target variance
        self.D = decorrelation #target dec dist
        self.corr_gauss = corr_gauss #target channel correlation
        self.gauss1 = []
        self.gauss2 = []
        #rayleigh
        self.N_rayleigh = 20 # summands
        self.var_rayleigh = variance_rayleigh #target variance
        self.D_ray = decorrelation / dec_factor # target dec dist
        self.corr_rayleigh = corr_rayleigh #target correlation
        self.ray1 = []
        self.ray2 = []
        #gesamt
        self.fading_1 = []
        self.fading_2 = []
        #Empirival values of the created arrays
        self.corr_g = -5
        self.corr_r = -5
        self.var_g1 = -5
        self.var_g2 = -5
        self.var_r1 = -5
        self.var_r2 = -5
        self.dec_g1 = 0
        self.dec_g2 = 0
        self.dec_r1 = 0
        self.dec_r2 = 0
        #path for saving plots
        self.path_plots = p_save


    def autocor_rayleigh(self, l, N, var, type="new", theta_arr=[], phi_real_arr=[], phi_imag_arr=[],
                         scale_theta=0.0):
        """
        create autocorrelated rayleigh array with SoS method.
        Two cases
        type = "new": New array is created --> random variables are created
                    return: created array and random variables
        type = "corr": Correlated array is created. Uniform distribution is added to RVs from initial array. RVs have to
                    given as input of the funktion
                    return: correlated array
        """
        # Copy input arrays
        phi_real_corr = deepcopy(phi_real_arr)
        phi_imag_corr = deepcopy(phi_imag_arr)
        theta_corr = deepcopy(theta_arr)
        if type == "new":
            phi_real_corr = []
            theta_corr = []

        # Calculate c for setting variance-->fading strength
        sigma_squared = 2/(4-np.pi) * var
        c = np.sqrt(2 * sigma_squared / N)


        # Create input in order to set the dec dist
        def linearFunc(y):
            return (y - 0.55444444)/0.2042099
        temp_fact = linearFunc(self.D_ray)
        x = np.tile(np.linspace(0, l/temp_fact, l), (self.N_rayleigh, 1))
        range_temp = np.tile(np.array([np.linspace(0, self.N_rayleigh - 1, self.N_rayleigh)]).transpose(), (1, self.l))
        ###
        if type == "new":
            phi_real = np.tile(np.array([np.random.uniform(-np.pi, np.pi, self.N_rayleigh)]).transpose(), (1, self.l))
            phi_imag = np.tile(np.array([np.random.uniform(-np.pi, np.pi, self.N_rayleigh)]).transpose(), (1, self.l))
            theta = np.random.uniform(-np.pi, np.pi, self.N_rayleigh)
            theta_calc = np.tile(np.array([theta]).transpose(), (1, self.l))
        if type == "corr":
            phi_real = phi_real_corr
            phi_imag = phi_imag_corr
            theta = np.random.uniform(np.array(theta_corr) - scale_theta, np.array(theta_corr) + scale_theta, size=self.N_rayleigh)
            theta_calc = np.tile(np.array([theta]).transpose(), (1, self.l))
        # calc alpha
        alpha = (2 * np.pi * range_temp - np.pi + theta_calc) / (4 * self.N_rayleigh)
        # Calculate real and imaginary part of one summand
        #sinusoid_real = np.zeros(self.N_rayleigh, self.l)
        sinusoid_real = c * np.cos(2 * np.pi * x * np.cos(alpha) + phi_real)
        sinusoid_imag = c * np.cos(2 * np.pi * x * np.sin(alpha) + phi_imag)

        # add elements
        h_real = sinusoid_real.sum(axis=0)
        h_imag = sinusoid_imag.sum(axis=0)

        # sqrt and add
        h_real_squared = h_real ** 2
        h_imag_squared = h_imag ** 2
        v = np.array(np.sqrt(np.add(h_real_squared, h_imag_squared)))

        # retrun arrays / random varibles
        if type == "new":
            return v, phi_real, phi_imag, theta
        if type == "corr":
            return v

    def correlated_rayleigh(self):
        """
        In this function initial and correlated rayleigh array are created and safed
        """
        scale = 0
        def exponential(x):
            #a* (np.e ** (-y / b)) + c
            a = 1.01368536
            b = 0.00626984
            c = 0.0775104
            return -b*np.log((x-c)/a)
        def linearFunc(y):
            return (y - 0.55444444)/0.2042099

        # Create initial array
        v, phi_real, phi_imag, theta = self.autocor_rayleigh(self.l, self.N_rayleigh, self.var_rayleigh, type="new")

        # Case: Target Correlation equals 1
        if self.corr_rayleigh == 1:
            self.ray1 = v
            self.ray2 = v
            self.corr_r = 1
            return

        # Calculate scale for normal distribution of correlated array
        if self.corr_rayleigh != 1 and self.corr_rayleigh >= 0.09:
            scale = exponential(self.corr_rayleigh)
        if self.corr_rayleigh < 0.09:
            scale = 20
        # Rescaling
        scale = scale * (linearFunc(self.D_ray)/100) * (1000000/self.l)

        # Calculate correalted array
        v_corr = []
        fehler = 1
        while fehler > 0.05:
            v_corr = self.autocor_rayleigh(self.l, self.N_rayleigh, self.var_rayleigh, type="corr", theta_arr=deepcopy(theta), phi_real_arr=deepcopy(phi_real),
                                      phi_imag_arr=deepcopy(phi_imag), scale_theta=deepcopy(scale))
            corr = np.corrcoef(v, v_corr)[0][1]
            fehler = np.absolute(corr - self.corr_rayleigh)
            #print("Fehler = " + str(fehler))
        # Save arrays
        self.corr_r = np.corrcoef(v, v_corr)[0][1]
        self.ray1 = v
        self.ray2 = v_corr
        print("Creating Rayleigh arrays is finished")

    def autocor_norm(self, l, N, var, D, type="new", theta_corr=[], scale_theta=0.0):
        """
        create autocorrelated normal array with SoS method.
        Two cases
        type = "new": New array is created --> random variables are created
                    return: created array and random variables
        type = "corr": Correlated array is created. Uniform distribution is added to RVs from initial array. RVs have to
                    given as input of the funktion
                    return: correlated array
        """
        if type == "new":
            theta_corr = []
        # Initialize array
        v = np.zeros(l)
        # Create Input array
        x = np.tile(np.linspace(0, l - 1, l), (self.N_gauss, 1))
        range_temp = np.tile(np.array([np.linspace(0, self.N_gauss - 1, self.N_gauss)]).transpose(), (1, self.l))
        # Set fading strength
        c = np.sqrt(2 * var / N)
        # Create random variables
        if type == "new":
            theta = np.random.uniform(0, 2 * np.pi, size=self.N_gauss)
            theta_calc = np.tile(np.array([theta]).transpose(), (1, self.l))
        if type == "corr":
            theta = np.random.uniform(theta_corr - scale_theta, theta_corr + scale_theta, size=self.N_gauss)
            theta_calc = np.tile(np.array([theta]).transpose(), (1, self.l))
        # Calc AoA
        alpha = (1 / (2 * np.pi * self.D)) * np.tan((np.pi * (range_temp - 0.5)) / (2 * self.N_gauss))
        #Calc Sinusoid
        sinusoid = c * np.cos(2 * np.pi * x * alpha + theta_calc)
        v = sinusoid.sum(axis=0)
        # return variables
        if type == "new":
            return v, theta
        if type == "corr":
            return v

    def correlated_gauss(self):
        """
        In this function initial and correlated gaussian array are created and safed
        """
        def cosinus(x):
            a = 0.59815972
            b = 0.73654715
            c = 0.39853684
            fehler = 1
            y = 0
            while fehler > 0.001:
                temp = a * np.cos(b * y) + c
                fehler = np.absolute(temp - x)
                y += 0.00001
            return y
        # Create initial array
        v, theta = self.autocor_norm(l=self.l, N=self.N_gauss, var=self.var_gauss, D=self.D, type="new")
        # Case: Target correlation = 1 --> arrays are duplicated
        if self.corr_gauss == 1:
            self.gauss1 = v
            self.gauss2 = v
            self.corr_g = 1
            return
        # Calculate scale
        if self.corr_gauss != 1 and self.corr_gauss != 0:
            scale = cosinus(self.corr_gauss)
        if self.corr_gauss == 0:
            scale = 20
        # Calculate correlated array
        v_corr = []
        fehler = 1
        while fehler > 0.05:
            v_corr = self.autocor_norm(l=self.l, N=self.N_gauss, var=self.var_gauss, D=self.D, theta_corr=theta,
                                       type="corr", scale_theta=scale)
            corr = np.corrcoef(v, v_corr)[0][1]
            fehler = np.absolute(corr - self.corr_gauss)
        # save values
        self.corr_g = np.corrcoef(v, v_corr)[0][1]
        self.gauss1 = v
        self.gauss2 = v_corr
        print("Creating Gauss arrays is finished")

    def create_fading(self):
        """
        In this function total fading is created from Gaussian and Rayleigh fading in dB
        """
        def linear_to_dB(data):
            arr = 10 * np.log10(data)
            return arr
        # Create Rayleigh/Gauss
        self.correlated_rayleigh()
        self.correlated_gauss()

        # Convert to db
        rayleigh1_dB = [linear_to_dB(self.ray1[i]) for i in range(self.l)]
        rayleigh2_dB = [linear_to_dB(self.ray2[i]) for i in range(self.l)]

        # Add small scale fading and shadow fading
        self.fading_1 = np.add(self.gauss1, rayleigh1_dB)
        self.fading_2 = np.add(self.gauss2, rayleigh2_dB)
        # Save and return total fading values
        return self.fading_1, self.fading_2

    def corr_var(self):
        """
        In this function the variances of the created arrays are calculated
        """
        self.var_g1 = (np.var(self.gauss1))
        self.var_g2 = (np.var(self.gauss2))
        self.var_r1 = (np.var(self.ray1))
        self.var_r2 = (np.var(self.ray2))
        return self.var_g1, self.var_g2, self.var_r1, self.var_r2, self.corr_g, self.corr_r

    def get_decor_dist(self):
        """
        In this method the decorrelation distance of the arrays is saved
        """
        def find1_e(v):
            num_lags = len(v) - 1
            autocor = sm.tsa.acf(v, nlags=num_lags)
            for i in range(len(autocor)):
                if autocor[i] <= 1/np.e:
                    return i
        self.dec_g1 = find1_e(self.gauss1)
        self.dec_g2 = find1_e(self.gauss2)
        self.dec_r1 = find1_e(self.ray1)
        self.dec_r2 = find1_e(self.ray2)
        return self.dec_g1, self.dec_g2, self.dec_r1, self.dec_r2

    def plot_rayleigh(self, v, savename):
        """
        In this function pdf and cdf of data and fit of normal dist is plotted
        """
        f_size_ax = 14
        f_size_tick = 12
        path = self.path_plots
        def rayleigh_cdf(x, loc, scale):
            return rayleigh.cdf(x, loc=loc, scale=scale)
        def rayleigh_pdf(x, loc, scale):
            return rayleigh.pdf(x, loc=loc, scale=scale)

        bin_no = np.linspace(np.amin(v), np.amax(v), len(v))
        data_entries, bins = np.histogram(v, bin_no)
        data_entries = data_entries / sum(data_entries)  # len or sum?
        cdf_data = np.cumsum(data_entries)
        bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
        popt, pcov = curve_fit(rayleigh_cdf, bincenters, cdf_data,
                               maxfev=1000000)  # , p0=[mu_shadowing, sigma_shadowing_factor])
        #print("popt= " + str(popt))
        plt.plot(bincenters, cdf_data, color="blue", label='Data of autocorrelated array')
        plt.plot(bincenters, rayleigh_cdf(bincenters, *popt), color='orange', linestyle="dotted", linewidth=3.0,
                 label='Rayleigh - fitted cumulated distribution function')
        plt.xlabel("Value")
        plt.ylabel("Cumulated Probability")
        plt.legend()
        # plt.savefig(path + str("Ray1_Rayleigh.pdf"))
        plt.savefig(path + savename + "_" + "Rayleigh_cdf.pdf")
        plt.close()

        data_entries, bins = np.histogram(v, 50, density=True)
        bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
        pdf_val = rayleigh_pdf(bincenters, *popt)
        plt.hist(v, bins=50, density=True)
        plt.plot(bincenters, pdf_val)
        plt.xlabel("Fading strength", fontsize=f_size_ax)
        plt.ylabel("Probability density", fontsize=f_size_ax)
        plt.xticks(fontsize=f_size_tick)
        plt.yticks(fontsize=f_size_tick)
        plt.grid()
        plt.savefig(self.path_plots + savename + "_" + "Rayleigh_pdf.pdf")
        tikzplotlib.save(self.path_plots + savename + "_" + "Rayleigh_pdf.pgf")
        plt.close()

    def plot_gauss(self, v, savename):
        """
        In this function pdf and cdf of data and fit of normal dist is plotted
        """
        f_size_ax = 14
        f_size_tick = 12
        path = self.path_plots
        def gauss_cdf(data, loc, scale):
            return norm.cdf(data, loc, scale)
        def gauss_pdf(data, loc, scale):
            return norm.pdf(data, loc, scale)
        # plot Norm CDF
        bin_no = np.linspace(np.amin(v), np.amax(v), len(v))
        data_entries, bins = np.histogram(v, bin_no)
        data_entries = data_entries / sum(data_entries)  # len or sum?
        cdf_data = np.cumsum(data_entries)
        bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
        popt, pcov = curve_fit(gauss_cdf, bincenters, cdf_data,
                               maxfev=1000000)  # , p0=[mu_shadowing, sigma_shadowing_factor])
        #print("popt= " + str(popt))
        plt.plot(bincenters, cdf_data, color="blue", label='Data of autocorrelated array')
        plt.plot(bincenters, gauss_cdf(bincenters, *popt), color='orange', linestyle="dotted", linewidth=3.0,
                 label='Normal - fitted cumulated distribution function')
        plt.xlabel("Value")
        plt.ylabel("Cumulated Probability")
        plt.legend()
        plt.savefig(self.path_plots + savename + "_" + "Gauss_cdf.pdf")
        plt.close()

        # Plot Lognormal PDF
        data_entries, bins = np.histogram(v, 50, density=True)
        bincenters = np.array([0.5 * (bins[i] + bins[i + 1]) for i in range(len(bins) - 1)])
        pdf_val = gauss_pdf(bincenters, *popt)
        plt.hist(v, bins=50, density=True)
        plt.plot(bincenters, pdf_val)
        plt.xlabel("Fading strength", fontsize=f_size_ax)
        plt.ylabel("Probability density", fontsize=f_size_ax)
        plt.xticks(fontsize=f_size_tick)
        plt.yticks(fontsize=f_size_tick)
        plt.tight_layout()
        plt.savefig(self.path_plots + savename + "_" + "Gauss_pdf.pdf")
        plt.close()

    def plot_autocor(self, v, num_lags, savename, target):
        """
        This method plots the autocorrelation function of "v" with "num_lags" lags and a certain name "savename"
        """
        f_size_ax = 14
        f_size_tick = 12
        num_lags = int(num_lags)
        x_lags = np.linspace(0, num_lags, num_lags + 1)
        autocor = sm.tsa.acf(v, nlags=num_lags)
        plt.plot(x_lags, autocor, label="Autocorrelation function")
        plt.xlabel("TTIs shifted", fontsize=f_size_ax)
        plt.ylabel("Correlation Coefficient", fontsize=f_size_ax)
        plt.ylim(ymax=1.2, ymin=-1)
        if target == "not_valid":
            pass
        else:
            plt.axvline(x=target, ymin=-1, ymax=1, color="lightgray", linestyle="--", label="TTI " + str(target))
        plt.axhline(y=0, color='lightgray')
        plt.xticks(fontsize=f_size_tick)
        plt.yticks(fontsize=f_size_tick)
        plt.tight_layout()
        plt.axhline(y=1 / np.e, color='gray', label="Correlation coefficient = 1/e")
        plt.legend()
        plt.savefig(self.path_plots + savename + "_" + "autocorrelation.pdf")
        #tikzplotlib.save(self.path_plots + savename + "_" + "ticks_autocorrelation.pgf")
        plt.close()

    def plot_fading_ausschnitt(self, v, extract, savename):
        """
        Fading array itself is plotted with respect to its index. Number of plotted indexes can be given with variable
        "extract"
        """
        x = np.linspace(0, extract - 1, extract)
        plt.plot(x, v[0:extract])
        plt.xlabel("TTI")
        plt.ylabel("Fading")
        plt.savefig(self.path_plots + savename + "_" + "fading.pdf")
        plt.close()

    def plot_qq(self, v, x_label, y_label, savename, ray):
        """
        By calling this function a QQ plot is created. There are two cases:
        ray = 1: Theoretical distribution is rayleigh
        ray = 0: Theoretical distribution is gauss
        """
        f_size_ax = 14
        f_size_tick = 12
        if ray == 1:
            sm.qqplot(np.array(v[0:2000]), dist=scipy.stats.rayleigh, fit=False,
                      line="q")
        if ray == 0:
            sm.qqplot(np.array(v), line="q")
        plt.xlabel(x_label, fontsize=f_size_ax)
        plt.ylabel(y_label, fontsize=f_size_ax)
        plt.xticks(fontsize=f_size_tick)
        plt.yticks(fontsize=f_size_tick)
        plt.tight_layout()
        plt.grid()
        plt.savefig(self.path_plots + savename + ".png")
        #tikzplotlib.save(self.path_plots + savename + ".pgf")
        plt.close()

    def plot_everything(self):
        """
        by calling this methods all plots are created and saved
        """
        # Gauss1
        self.plot_gauss(self.gauss1, "gauss1")
        self.plot_autocor(self.gauss1, 2*self.D, "gauss1", self.D)
        # Gauss2
        self.plot_gauss(self.gauss2, "gauss2")
        self.plot_autocor(self.gauss2, 2 * self.D, "gauss2", self.D)
        # Rayleigh 1
        self.plot_rayleigh(self.ray1, "ray1")
        self.plot_autocor(self.ray1, 15 * (self.D/10), "ray1", self.D_ray)
        # Rayleigh 2
        self.plot_rayleigh(self.ray2, "ray2")
        self.plot_autocor(self.ray2, 15 * (self.D / 10), "ray2", self.D_ray)
        # Plot Fading
        self.plot_fading_ausschnitt(self.fading_1, 8*self.D, "total")
        self.plot_fading_ausschnitt(self.gauss1, 8 * self.D, "gauss")
        self.plot_fading_ausschnitt(self.ray1, 8 * self.D, "Rayleigh")
        self.plot_autocor(self.fading_1, 8 * self.D, "Total", "not_valid")
        # Plot qq
        self.plot_qq(self.gauss1, "Theoretical normalized quantiles", "Empirical normalized quantils", "gauss1_qq", ray=0)
        self.plot_qq(self.gauss2, "Theoretical normalized quantiles", "Empirical normalized quantils", "gauss2_qq", ray=0)
        self.plot_qq(self.ray1, "Theoretical Quantiles", "Empirical Quantils", "ray1_qq", ray=1)
        self.plot_qq(self.ray2, "Theoretical Quantiles", "Empirical Quantils", "ray2_qq", ray=1)

    #def log_file(self):

#### Example ####
# Create instance of fading class
"""
Parameters:
corr_gauss: Correlation of two gaussian arrays for large scale fading
corr_rayleigh: correlation of two rayleigh arrays for small scale fading
variance_gauss: 
variance_rayleigh:
decorrelation: decorrelation distance small scale fading
dec_factor: factor betwenn large scale and small scale dec dist; e.g decorrelation distance large scale fading: 1000
            and dec_factor= 10, then: dec dist small scale fading = 100
"""
start = timeit.default_timer()



fad = Fading(number_values=1000000, corr_gauss=0.1,
                 corr_rayleigh=0.1, variance_rayleigh=5,
                 variance_gauss=25, decorrelation=2000, dec_factor=10, p_save="TODO: insert path to directory for saving plots")

## MAIN ##
# call method to create fading array
x,y = fad.create_fading()
end = timeit.default_timer()
print("Time to create fading arrays: " + str(round(end - start, 2)) + "s")
# plot important information
fad.plot_everything()
# get variance of created arrays
v_g1, v_g2, v_r1, v_r2, corr_g, corr_r = fad.corr_var()
print("Variance of first gaussian array: " + str(v_g1))
print("Variance of second gaussian array: " + str(v_g2))
print("Variance of first rayleigh array: " + str(v_r1))
print("Variance of second rayleigh array: " + str(v_r2))
print("Correlation coeff of gaussian arrays: " + str(corr_g))
print("Correlation coeff of rayleigh arrays: " + str(corr_r))

