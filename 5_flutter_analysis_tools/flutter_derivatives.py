
import numpy as np


class FlutterDerivatives():

    def __init__(self, default_notation='real'):

        # Notation default settings
        self.available_notations = ['real', 'complex']
        self.check_notation(default_notation)
        self.default_notation = default_notation
        
        # Initializing dictionaries with the derivatives' data
        self.real_data_struct = {'values':[], 'U_red':[]}
        self.complex_data_struct = {'values':[],'k':[]}
        self.reset_all_derivatives()


    def reset_all_derivatives(self):
        
        # Creating/Overwriting derivatives dictionary (real notation)
        self.fd_real = {}
        for letter in ['H', 'A', 'P']:
            for number in range(1,7):
                self.fd_real[letter+str(number)] = self.real_data_struct

        # Creating/Overwriting derivatives dictionary (complex notation)
        self.fd_complex = {}
        for letter_1 in ['h', 'a', 'p']:
            for letter_2 in ['h', 'a', 'p']:
                self.fd_complex[letter_1+letter_2] = self.complex_data_struct


    def reset_derivative(self, deriv):

        self.check_derivative_name(deriv)
        
        if deriv in list(self.fd_real.keys()):
            self.fd_real[deriv] = self.real_data_struct
        elif deriv in list(self.fd_complex.keys()):
            self.fd_complex[deriv] = self.complex_data_struct

    
    def reset_from_dictionary(self, dict):
        pass

    
    def check_derivative_name(self, deriv):

        real_deriv_names = list(self.fd_real.keys())
        complex_deriv_names = list(self.fd_complex.keys())
            
        if deriv not in real_deriv_names or deriv not in complex_deriv_names:
            msg = 'Derivative name not recognised. '
            msg += 'It should be one of the following:\n'
            msg += str(real_deriv_names+complex_deriv_names)
            raise Exception(msg)


    def check_notation(self, notation):
        
        if notation not in self.available_notations:
            msg = 'The requested notation is not among the available ones.'
            msg += ' Please select one of these: ' + str(self.available_notations)
            raise Exception(msg)

    
    def change_default_notation(self, new_notation):

        self.check_notation(new_notation)
        self.default_notation = new_notation


    def get_all_derivatives(self, notation='default'):

        if notation == 'default':
            notation = self.default_notation

        self.check_notation(notation)
        
        if notation == 'real':
            return self.fd_real

        elif notation == 'complex':
            return self.fd_complex


    def get_derivative(self, deriv):

        self.check_derivative_name(deriv)
        
        if deriv in self.fd_real.keys():
            return self.fd_real[deriv]['values'], self.fd_real[deriv]['U_red']

        elif deriv in self.fd_complex.keys():
            return self.fd_real[deriv]['values'], self.fd_real[deriv]['k']

    
    def set_default_parameters(self, U=None, B=None, delta_t=None):

        if U != None:
            self.U = U
        
        if B != None:
            self.B = B
        
        if delta_t != None:
            self.delta_t = delta_t

    
    def calculate_derivatives_from_forced_motion(self,
                    heave=None, pitch=None, sway=None,
                    lift=None, moment=None, drag=None,
                    U=None, B=None, delta_t=None):
        
        # Start checking simulation parameters
        U, B, delta_t = self.fill_with_default_simulation_parameters(U, B, delta_t)

    def calculate_single_derivative_pair_from_forced_motion(self):

        pass


    def fill_with_default_simulation_parameters(self, U, B, delta_t):

        msg = 'The variable {} has no default value. It is necessary '
        msg = 'to provide a particular one when calling the function.'

        if U == None:
            if self.U != None:
                U = self.U
            else:
                raise Exception(msg.format('U'))
        if B == None:
            if self.B != None:
                B = self.B
            else:
                raise Exception(msg.format('B'))
        if delta_t == None:
            if self.delta_t != None:
                delta_t = self.delta_t
            else:
                raise Exception(msg.format('delta_t'))

        return(U, B, delta_t)

        
def complex2real_notation(fd_complex):

    fd_real = {'H1':{'values':[],'U_red':[]},
        'H2':{'values':[],'U_red':[]},
        'H3':{'values':[],'U_red':[]},
        'H4':{'values':[],'U_red':[]},
        'A1':{'values':[],'U_red':[]},
        'A2':{'values':[],'U_red':[]},
        'A3':{'values':[],'U_red':[]},
        'A4':{'values':[],'U_red':[]}}
    
    for letter_1 in 'ha':

        if letter_1 == 'a':
            factor_1 = 0.5
            deriv_letter = 'A'
        else:
            factor_1 = 1
            deriv_letter = 'H'

        for letter_2 in 'ha':
            
            if letter_2 == 'a':
                factor_2 = 0.5
                deriv_numbers = [2,3]
            else:
                factor_2 = 1
                deriv_numbers = [1,4]

            for input_value, k in zip(fd_complex['c_'+letter_1+letter_2]['values'], fd_complex['c_'+letter_1+letter_2]['k']):

                U_red = np.pi / k

                output_values = np.array([np.imag(input_value), np.real(input_value)]) * np.pi * 0.5 * factor_1 * factor_2

                for output_value, deriv_number in zip(output_values, deriv_numbers):
                    fd_real[deriv_letter+str(deriv_number)]['values'].append(output_value)
                    fd_real[deriv_letter+str(deriv_number)]['U_red'].append(U_red)
    
    return fd_real
                

def real2complex_notation(fd_real):
    pass


if __name__ == '__main__':
    fd_complex = {'c_aa':{'values':[2.4-0.36j],'k':[0.5]},
        'c_ah':{'values':[0.0071+1.2j],'k':[0.5]},
        'c_ha':{'values':[-3.6-0.45j],'k':[0.5]},
        'c_hh':{'values':[0.063-1.6j],'k':[0.5]}}

    fd_real = complex2real_notation(fd_complex)
    print(fd_real)