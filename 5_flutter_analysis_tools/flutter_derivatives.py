
import numpy as np
import sinusoidal_utilities as sin_util
import warnings


# Main object to manage flutter derivatives data
class FlutterDerivatives():

    def __init__(self, default_notation='real'):

        # Notation default settings
        # 'real' notation --> Scanlan's notation (H1, H2, A1, A2...)
        # 'complex' notation --> Starossek's notation (c_hh, c_ha...)
        self.available_notations = ['real', 'complex']
        self._check_notation(default_notation)
        self.default_notation = default_notation
        
        # Initializing dictionaries with the derivatives' data
        self.reset_all_derivatives()


    def reset_all_derivatives(self):

        # Dictionary to store all derivative values, in all notations
        self.flutter_deriv = {'real':{}, 'complex':{}}
        
        # Creating/Overwriting derivatives dictionary (real notation)
        self.flutter_deriv['real'] = {}
        for letter in ['H', 'A', 'P']:
            for number in range(1,7):
                self.flutter_deriv['real'][letter+str(number)] = {'values':[], 'Ured':[]}

        # Creating/Overwriting derivatives dictionary (complex notation)
        self.flutter_deriv['complex'] = {}
        for letter_1 in ['h', 'a', 'p']:
            for letter_2 in ['h', 'a', 'p']:
                self.flutter_deriv['complex']['c_'+letter_1+letter_2] = {'values':[],'k':[]}


    def reset_derivative(self, deriv):

        # Check that the derivative's name is among the available ones
        self._check_derivative_name(deriv)
        
        # Clean the derivative data
        # TODO: clean associated derivatives in other notations
        # Maybe the derivatives need to be cleaned by pairs (H1-H4, A2-A3...)
        # If not, it is not possible to clean complex notation derivatives
        if deriv in list(self.flutter_deriv['real'].keys()):
            self.flutter_deriv['real'][deriv] = {'values':[], 'Ured':[]}
        elif deriv in list(self.flutter_deriv['complex'].keys()):
            self.flutter_deriv['complex'][deriv] = {'values':[],'k':[]}

    
    def reset_from_dictionary(self, dict):

        # TODO: Function to start a FlutterDerivatives object
        # from a self.flutter_deriv-type dictionary.
        # Necessary to check right format in input dictionary.

        pass

    
    def _check_derivative_name(self, deriv):

        # Listing all available derivative names
        real_deriv_names = list(self.flutter_deriv['real'].keys())
        complex_deriv_names = list(self.flutter_deriv['complex'].keys())
        
        # Raise exception if the input is not among the available ones
        if deriv not in real_deriv_names and deriv not in complex_deriv_names:
            msg = 'Derivative name not recognised. '
            msg += 'It should be one of the following:\n'
            msg += str(real_deriv_names+complex_deriv_names)
            raise Exception(msg)


    def _check_notation(self, notation):
        
        # Raise exception if the input is not among the available ones
        if notation not in self.available_notations:
            msg = 'The requested notation is not among the available ones.'
            msg += ' Please select one of these: ' + str(self.available_notations)
            raise Exception(msg)

    
    def change_default_notation(self, new_notation):
        
        # Change output default notation
        self._check_notation(new_notation)
        self.default_notation = new_notation


    def get_all_derivatives(self, notation=None):

        # Use default notation if none was provied
        if notation == None:
            notation = self.default_notation
        # If not, check the input notation
        else:
            self._check_notation(notation)
        
        # Return a dictionary with all the derivative data
        return self.flutter_deriv[notation]


    def get_derivative(self, deriv):

        # Check if input is among the available ones
        self._check_derivative_name(deriv)
        
        # Return a small dictionary with only data from one derivative
        if deriv in self.flutter_deriv['real'].keys():
            return self.flutter_deriv['real'][deriv]
        elif deriv in self.flutter_deriv['complex'].keys():
            return self.flutter_deriv['complex'][deriv]

    
    def set_default_parameters(self, U=None, B=None, delta_t=None, air_dens=None):

        # TODO: check input format
        # Maybe also use **kwargs instead

        if U != None:
            self.U = U
        
        if B != None:
            self.B = B
        
        if delta_t != None:
            self.delta_t = delta_t

        if air_dens != None:
            self.air_dens = air_dens

    
    def calculate_derivatives_from_forced_motion(self, **kwargs):
        
        # Start checking simulation parameters
        # TODO: consider simply changing the kwargs dictionary with the filled parameters
        sim_params = self._fill_with_default_simulation_parameters(**kwargs)

        # Check that the parameters have frequency data.
        # If yes, add it to the sim_params dictionary
        # I not, add 'None' to the dictionary
        sim_params['omega'] = self._check_frequency_input(**kwargs)

        # Check that the right motion and force time series have been provided
        provided_motion, provided_forces = self._check_motion_force_input(**kwargs)

        # Calculating derivatives pair by pair (with the motion and one force)
        for provided_force in provided_forces:
            self._calculate_derivative_pair_from_forced_motion(sim_params, provided_motion, provided_force, kwargs)
        

    def _calculate_derivative_pair_from_forced_motion(self, sim_params, m_name, f_name, data):

        # Extract the corresponding motion and force time-series
        motion = data[m_name]
        force = data[f_name]

        # Create the time time-series
        # It is just necessary to respect the time step
        # (the absolute value is not important, it will be shifted later). 
        time = [i*sim_params['delta_t'] for i in range(len(motion))]

        # Calculate the amplitude and phase of the motion
        if sim_params['omega'] == None:
            motion_ampl, phi, omega = sin_util.extract_sinusoidal_parameters(time, motion)
        else:
            omega = sim_params['omega']
            motion_ampl, phi = sin_util.extract_sinusoidal_parameters(time, motion, omega=omega)

        # Shift time series to ensure that the motion has no phase
        # (just a pure sinusoidal).
        time_lag = phi/omega
        time += time_lag

        # Fit the sine+cosine function
        # force = a + b*cos(omega*t) + c*sin(omega*t)
        a, b, c = sin_util.extract_sinusoidal_parameters(time, force, omega=omega, function='sin_cos')

        # Get the names of the derivatives asociated
        # with this particular motion and force directions
        derivs_to_calc = self._get_derivatives_to_calculate(m_name, f_name)

        # Calculation of the flutter derivatives
        # COMPLEX NOTATION
        #      parameter            factor_0                factor_1    factor_2
        # H1       b     *   2/(dens*B^2*omega^2*ampl)   *     1      *    1
        # H4       c     *   2/(dens*B^2*omega^2*ampl)   *     1      *    1
        # A1       b     *   2/(dens*B^2*omega^2*ampl)   *     1      *   1/B
        # A4       c     *   2/(dens*B^2*omega^2*ampl)   *     1      *   1/B
        # H2       b     *   2/(dens*B^2*omega^2*ampl)   *    1/B     *    1
        # H3       c     *   2/(dens*B^2*omega^2*ampl)   *    1/B     *    1
        # A2       b     *   2/(dens*B^2*omega^2*ampl)   *    1/B     *   1/B
        # A3       c     *   2/(dens*B^2*omega^2*ampl)   *    1/B     *   1/B
        # REAL NOTATION
        #      parameter                factor_0               factor_1    factor_2
        # H1     c+b*i   *   4/(dens*B^2*omega^2*ampl*pi)   *     1      *    1
        # A1     c+b*i   *   4/(dens*B^2*omega^2*ampl*pi)   *     1      *   2/B
        # H2     c+b*i   *   4/(dens*B^2*omega^2*ampl*pi)   *    2/B     *    1
        # A2     c+b*i   *   4/(dens*B^2*omega^2*ampl*pi)   *    2/B     *   2/B
        f0_real = 2/(sim_params['air_dens']*omega**2*sim_params['B']**2*motion_ampl)
        f0_complex = 2*f0_real/np.pi
        if m_name == 'pitch':
            f1_real = 1/sim_params['B']
            f1_complex = 2*f1_real
        else:
            f1_real = 1
            f1_complex = 1
        if f_name == 'moment':
            f2_real = 1/sim_params['B']
            f2_complex = 2*f1_real
        else:
            f2_real = 1
            f2_complex = 1
        
        # Calculating and saving derivative values
        self.flutter_deriv['real'][derivs_to_calc['real'][0]]['values'].append(b*f0_real*f1_real*f2_real)
        self.flutter_deriv['real'][derivs_to_calc['real'][1]]['values'].append(c*f0_real*f1_real*f2_real)
        self.flutter_deriv['complex'][derivs_to_calc['complex']]['values'].append(complex(c,b)*f0_complex*f1_complex*f2_complex)

        # Calculating 'x-axis' values (the derivatives depend on them)
        freq = omega/2/np.pi
        Ured = sim_params['U']/freq/sim_params['B']
        k = omega*sim_params['B']/2/sim_params['U']

        # Saving 'x-axis' data
        self.flutter_deriv['real'][derivs_to_calc['real'][0]]['Ured'].append(Ured)
        self.flutter_deriv['real'][derivs_to_calc['real'][1]]['Ured'].append(Ured)
        self.flutter_deriv['complex'][derivs_to_calc['complex']]['k'].append(k)


    def _fill_with_default_simulation_parameters(self, **kwargs):
        
        # Preparation of the error message
        msg = 'The variable "{}" has no default value. It is necessary '
        msg += 'to provide a particular one when calling the function.'

        # Check if there is a provided value
        # If not, check if there is at least a default value
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

        if 'air_dens' in kwargs:
            air_dens = kwargs['air_dens']
        else:
            if getattr(self, 'air_dens', None) != None:
                air_dens = self.air_dens
            else:
                raise Exception(msg.format('air_dens'))
        
        # Return input parameters filled with default values when necessary
        sim_params = {'U':U, 'B':B, 'delta_t':delta_t, 'air_dens':air_dens}
        return(sim_params)


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

        # TODO: check that all time series have the same length

        return list(provided_motion)[0], list(provided_forces)


    def _check_frequency_input(self, **kwargs):
        
        # Inputs in **kwargs that would give frequency data
        possible_frequency_inputs = {'omega', 'frequency'}
        provided_inputs = possible_frequency_inputs.intersection(kwargs)

        # If only one is given, calculate omega (the angular frequency),
        # which will be the one used later.
        if len(provided_inputs) == 1:
            provided_input = list(provided_inputs)[0]
            if provided_input == 'omega':
                omega = kwargs['omega']
            elif provided_input == 'frequency':
                omega = kwargs['frequency']*2*np.pi

        # If none was given, raise exception and ask for frequency data
        elif len(provided_inputs) == 0:
            msg = 'No motion frequency was provided. '
            msg += 'The frequency will be estimated from the motion time series instead. '
            msg += 'However, the following results may be compromised, '
            msg += 'even if the motion frequency is below the nyquist frequency.'
            warnings.warn(msg)
            omega = None
        
        # If more than one were given, raise exception
        # to avoid incoherences between them
        else:
            msg = 'More than one frequency inputs were given. '
            msg += 'In order to avoid contradictions provide '
            msg += 'only one of the following inputs: '
            msg += str(possible_frequency_inputs)
            raise Exception(msg)
        
        # Return the omega value, it will be used later
        return omega


    def _get_derivatives_to_calculate(self, m_name, f_name):
        # Each combination between motion (heave, pitch, sway) and
        # force (lift, moment, drag) has some derivatives associated
        # This function returns the corresponding derivative names

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
        
        # Bulding and returning dictionary with the derivative names
        derivs_to_calc = {}
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