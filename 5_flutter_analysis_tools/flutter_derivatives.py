
import numpy as np


class FlutterDerivatives():

    def __init__(self, default_notation='real'):

        # Notation default settings
        self.available_notations = ['real', 'complex']
        self._check_notation(default_notation)
        self.default_notation = default_notation
        
        # Initializing dictionaries with the derivatives' data
        self.real_data_struct = {'values':[], 'U_red':[]}
        self.complex_data_struct = {'values':[],'k':[]}
        self.reset_all_derivatives()


    def reset_all_derivatives(self):

        flutter_deriv = {'real':{}, 'complex':{}}
        
        # Creating/Overwriting derivatives dictionary (real notation)
        self.flutter_deriv['real'] = {}
        for letter in ['H', 'A', 'P']:
            for number in range(1,7):
                self.flutter_deriv['real'][letter+str(number)] = self.real_data_struct

        # Creating/Overwriting derivatives dictionary (complex notation)
        self.flutter_deriv['complex'] = {}
        for letter_1 in ['h', 'a', 'p']:
            for letter_2 in ['h', 'a', 'p']:
                self.flutter_deriv['complex']['c_'+letter_1+letter_2] = self.complex_data_struct


    def reset_derivative(self, deriv):

        self._check_derivative_name(deriv)
        
        if deriv in list(self.flutter_deriv['real'].keys()):
            self.flutter_deriv['real'][deriv] = self.real_data_struct
        elif deriv in list(self.flutter_deriv['complex'].keys()):
            self.flutter_deriv['complex'][deriv] = self.complex_data_struct

    
    def reset_from_dictionary(self, dict):
        pass

    
    def _check_derivative_name(self, deriv):

        real_deriv_names = list(self.flutter_deriv['real'].keys())
        complex_deriv_names = list(self.flutter_deriv['complex'].keys())
            
        if deriv not in real_deriv_names or deriv not in complex_deriv_names:
            msg = 'Derivative name not recognised. '
            msg += 'It should be one of the following:\n'
            msg += str(real_deriv_names+complex_deriv_names)
            raise Exception(msg)


    def _check_notation(self, notation):
        
        if notation not in self.available_notations:
            msg = 'The requested notation is not among the available ones.'
            msg += ' Please select one of these: ' + str(self.available_notations)
            raise Exception(msg)

    
    def change_default_notation(self, new_notation):

        self._check_notation(new_notation)
        self.default_notation = new_notation


    def get_all_derivatives(self, notation='default'):

        if notation == 'default':
            notation = self.default_notation

        self._check_notation(notation)
        
        return self.flutter_deriv[notation]


    def get_derivative(self, deriv):

        self._check_derivative_name(deriv)
        
        if deriv in self.flutter_deriv['real'].keys():
            return self.flutter_deriv['real'][deriv]['values'], self.flutter_deriv['real'][deriv]['U_red']

        elif deriv in self.flutter_deriv['complex'].keys():
            return self.flutter_deriv['complex'][deriv]['values'], self.flutter_deriv['complex'][deriv]['k']

    
    def set_default_parameters(self, U=None, B=None, delta_t=None):

        if U != None:
            self.U = U
        
        if B != None:
            self.B = B
        
        if delta_t != None:
            self.delta_t = delta_t

    
    def calculate_derivatives_from_forced_motion(self, **kwargs):
                    #heave=None, pitch=None, sway=None,
                    #lift=None, moment=None, drag=None,
                    #U=None, B=None, delta_t=None):
        
        # Start checking simulation parameters
        sim_params = self._fill_with_default_simulation_parameters(**kwargs)

        # Check that the right motion and force time series have been provided
        provided_motion, provided_forces = self._check_motion_force_input(**kwargs)

        # Calculating derivatives pair by pair (with the motion and one force)
        for force_name in provided_forces:
            self._calculate_derivative_pair_from_forced_motion(sim_params, provided_motion, force_name, kwargs)
        

    def _calculate_derivative_pair_from_forced_motion(self, sim_params, m_name, f_name, data):

        motion = data[m_name]
        force = data[f_name]

        print(m_name, f_name)


    def _fill_with_default_simulation_parameters(self, **kwargs):

        msg = 'The variable "{}" has no default value. It is necessary '
        msg += 'to provide a particular one when calling the function.'

        if 'U' in kwargs:
            U = kwargs['U']
        else:
            if getattr(self, 'U', None) != None:
                U = self.U
            else:
                raise Exception(msg.format('U'))

        if 'B' in kwargs:
            B = kwargs['B']
        else:
            if getattr(self, 'B', None) != None:
                B = self.B
            else:
                raise Exception(msg.format('B'))

        if 'delta_t' in kwargs:
            delta_t = kwargs['delta_t']
        else:
            if getattr(self, 'delta_t', None) != None:
                delta_t = self.delta_t
            else:
                raise Exception(msg.format('delta_t'))

        return(U, B, delta_t)


    def _check_motion_force_input(self, **kwargs):

        # Check that only one motion is provided
        motion_names = {'heave', 'pitch', 'sway'}
        provided_motion = motion_names.intersection(kwargs)
        if len(provided_motion) == 0:
            msg = 'No motion time series provided. '
            msg += 'Please provide one of the following variables: '
            msg += str(motion_names)
            raise Exception(msg)
        elif len(provided_motion) > 1:
            # TODO: This 'elif' would need to be suppressed if the function
            # is adapted to calculate from multi-direction simulations
            msg = 'Too many motion time series provided. '
            msg += 'Please provide only the excited motion time series.'
            raise Exception(msg)

        # Check that at least one force is provided
        force_names = {'lift', 'moment', 'drag'}
        provided_forces = force_names.intersection(kwargs)
        if len(provided_forces) == 0:
            msg = 'No force time series provided. '
            msg += 'Please provide at least one of the following variables: '
            msg += str(force_names)
            raise Exception(msg)

        return list(provided_motion)[0], list(provided_forces)


        def _get_derivatives_to_calculate(m_name, f_name):
            
            derivs_to_calc = {}

            if m_name == 'heave':
                real_indexes = [1,4]
                complex_letter_2 = 'h'
            elif m_name == 'pitch':
                real_indexes = [2,3]
                complex_letter_2 = 'a'
            elif m_name == 'sway':
                real_indexes = [5,6]
                complex_letter_2 = 'p'

            if f_name == 'lift':
                real_letter = 'H'
                complex_letter_1 = 'h'
            elif f_name == 'moment':
                real_letter = 'A'
                complex_letter_1 = 'a'
            elif f_name == 'drag':
                real_letter = 'P'
                if m_name == 'heave':
                    real_indexes = [5,6]
                elif m_name == 'sway':
                    real_indexes = [1,4]
                complex_letter_1 = 'p'
            
            derivs_to_calc['real'] = [real_letter+str(real_indexes[0]), real_letter+str(real_indexes[1])]
            derivs_to_calc['complex'] = 'c_' + complex_letter_1 + complex_letter_2

            return derivs_to_calc

            
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