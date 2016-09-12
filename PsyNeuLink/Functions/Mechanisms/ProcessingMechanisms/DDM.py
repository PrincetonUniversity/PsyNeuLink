# Princeton University licenses this file to You under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.  You may obtain a copy of the License at:
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software distributed under the License is distributed
# on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and limitations under the License.
#
#
# ***************************************************  DDM *************************************************************
#

# from numpy import sqrt, random, abs, tanh, exp
from PsyNeuLink.Functions.Mechanisms.ProcessingMechanisms.ProcessingMechanism import *
from PsyNeuLink.Functions.Utilities.Utility import *

# DDM outputs (used to create and name outputStates):
kwDDM_DecisionVariable = "DDM_DecisionVariable"
kwDDM_Error_Rate = "DDM_Error_Rate"
kwDDM_Probability_upperBound = "DDM_Probability_upperBound"
kwDDM_Probability_lowerBound = "DDM_Probability_lowerBound"
kwDDM_RT_Mean = "DDM_RT_Mean"
kwDDM_RT_Correct_Mean = "DDM_RT_Correct_Mean"
kwDDM_RT_Correct_Variance = "DDM_RT_Correct_Variance"
kwDDM_Total_Allocation = "DDM_Total_Allocation"
kwDDM_Total_Cost = "DDM_Total_Cost"


# Indices for results used in return value tuple; auto-numbered to insure sequentiality
class DDM_Output(AutoNumber):
    DECISION_VARIABLE = ()
    RT_MEAN = ()
    ER_MEAN = ()
    P_UPPER_MEAN = ()
    P_LOWER_MEAN = ()
    RT_CORRECT_MEAN = ()
    RT_CORRECT_VARIANCE = ()
    TOTAL_COST = ()
    TOTAL_ALLOCATION = ()
    NUM_OUTPUT_VALUES = ()


class DDMError(Exception):
    def __init__(self, error_value):
        self.error_value = error_value

    def __str__(self):
        return repr(self.error_value)


class DDM(ProcessingMechanism_Base):
# DOCUMENT:   COMBINE WITH INITIALIZATION WITH PARAMETERS
#                    ADD INFO ABOUT B VS. N&F
#                    ADD instantiate_output_states TO INSTANCE METHODS, AND EXPLAIN RE: NUM OUTPUT VALUES FOR B VS. N&F
    """Implement DDM subclass

    Description:
        DDM is a subclass Type of the Mechanism Category of the Function class
        It implements a Mechanism for several forms of the Drift Diffusion Model (DDM) for
            two alternative forced choice (2AFC) decision making:
            - Bogacz et al. (2006) analytic solution (see kwDDM_BogaczEtAl option below):
                generates error rate (ER) and decision time (DT);
                ER is used to stochastically generate a decision outcome (+ or - valued) on every run
            - Navarro and Fuss (2009) analytic solution (see kwDDM_NavarroAndFuss:
                generates error rate (ER), decision time (DT) and their distributions;
                ER is used to stochastically generate a decision outcome (+ or - valued) on every run
            [TBI: - stepwise integrator that simulates each step of the integration process

    Instantiation:
        - A DDM mechanism can be instantiated in several ways:
            - directly, by calling DDM()
            - as the default mechanism (by calling mechanism())

    Initialization arguments:
        In addition to standard arguments params (see Mechanism), DDM also implements the following params:
        - params (dict):
            + kwDDM_AnalyticSolution (str keyword):
                specifies analytic solution of the DDM to use;
                value must also be one of the following keywords:
                + kwDDM_BogaczEtAl: generates mean reaction time (RT) and mean error rate (ER) as described in:
                    Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal
                        decision making: a formal analysis of models of performance in two-alternative forced-choice
                        tasks.  Psychological review, 113(4), 700.
                    Notes:
                    * requires that DDM.execute be called with time_scale = TimeScale.TRIAL
                + kwDDM_NavarroAndFuss:  gives mean RT and ER as well as distributions as described in:
                    Navarro, D. J., and Fuss, I. G. "Fast and accurate calculations for first-passage times in
                        Wiener diffusion models." Journal of Mathematical Psychology 53.4 (2009): 222-230.
                    Notes:
                    * requires that matLab engine be installed
                    * requires that DDM.execute be called with time_scale = TimeScale.TRIAL
                [TBI: + kwIntegrate: executes step-wise intregation process, one step per CentralClock.time_step
                    Notes:
                    * requires that matLab engine be installed
                    * requires that DDM.execute be called with time_scale = TimeScale.REAL_TIME]
            + FUNCTION_PARAMS (dict):
                + kwDDM_DriftRate (float):
                    specifies internal ("attentional") component of the drift rate
                    that is added to the input (self.variable) on every call to DDM.execute()
                + kwDDM_StartingPoint (float):
                    specifies intitial value of decision variable, converted to "bias" term in DDM
                + kwDDM_Threshold (float):
                    specifies stopping value of decision variable for integration process
                + kwDDM_Noise (float):
                    specifies internal noise term for integration process
                + kwDDM_T0 (float):
                    specifies non-decision time added to total response time
        Notes:
        *  params can be set in the standard way for any Function subclass:
            - params provided in param_defaults at initialization will be assigned as paramInstanceDefaults
                 and used for paramsCurrent unless and until the latter are changed in a function call
            - paramInstanceDefaults can be later modified using assign_defaults
            - params provided in a function call (to execute or adjust) will be assigned to paramsCurrent

    MechanismRegistry:
        All instances of DDM are registered in MechanismRegistry, which maintains an entry for the subclass,
          a count for all instances of it, and a dictionary of those instances

    Naming:
        Instances of DDM can be named explicitly (using the name='<name>' argument).
        If this argument is omitted, it will be assigned "DDM" with a hyphenated, indexed suffix ('DDM-n')

    Execution:
        - Calculates either:
            analytic solutions:  estimated outcome for a run of the integration process (time_scale = TimeScale.TRIAL)
            integration process: step-wise trajectory of the integration process (time_scale = TimeScale.REAL_TIME)
        - self.value (and values of outputStates) contain each outcome value (e.g., ER, DT, etc.)
        - self.execute returns self.value
        Notes:
        * DDM handles "runtime" parameters (specified in call to execute method) differently than standard Functions:
            any specified params are kept separate from paramsCurrent (Which are not overridden)
            if the FUNCTION_RUN_TIME_PARMS option is set, they are added to the current value of the
                corresponding ParameterState;  that is, they are combined additively with controlSignal output

    Class attributes:
        + functionType (str): DDM
        + classPreference (PreferenceSet): DDM_PreferenceSet, instantiated in __init__()
        + classPreferenceLevel (PreferenceLevel): PreferenceLevel.TYPE
        + variableClassDefault (value):  DDM_Defaults.starting_point
        + paramClassDefaults (dict): {kwTimeScale: TimeScale.TRIAL,
                                      kwDDM_AnalyticSolution: kwDDM_BogaczEtAl,
                                      FUNCTION_PARAMS: {kwDDM_DriftRate:<>
                                                              kwDDM_StartingPoint:<>
                                                              kwDDM_Threshold:<>
                                                              kwDDM_Noise:<>
                                                              kwDDM_T0:<>},
                                      kwOutputStates: [kwDDM_DecisionVariable,
                                                       kwDDM_Error_Rate,
                                                       kwDDM_Probability_upperBound,
                                                       kwDDM_Probability_lowerBound,
                                                       kwDDM_RT_Mean,
                                                       kwDDM_RT_Correct_Mean,
                                                       kwDDM_RT_Correct_Variance,
                                                       kwDDM_Total_Allocation,
                                                       kwDDM_Total_Cost],
        + paramNames (dict): names as above

    Class methods:
        None

    Instance attributes: none
        + variable (value) - input to mechanism's execute method (default:  DDM_Defaults.starting_point)
        + value (value) - output of execute method
        + name (str) - if it is not specified as an arg, a default based on the class is assigned in register_category
        + prefs (PreferenceSet) - if not specified as an arg, a default set is created by copying DDM_PreferenceSet

    Instance methods:
        - instantiate_function(context)
            deletes params not in use, in order to restrict outputStates to those that are computed for specified params
        - execute(variable, time_scale, params, context)
            executes specified version of DDM and returns outcome values (in self.value and values of self.outputStates)
        - ou_update(particle, drift, noise, time_step_size, decay)
            single update for OU (special case l=0 is DDM) -- from Michael Shvartsman
        - ddm_update(particle, a, s, dt)
            DOCUMENTATION NEEDED
            from Michael Shvartsman
        - ddm_rt(x0, t0, a, s, z, dt)
            DOCUMENTATION NEEDED
            from Michael Shvartsman
        - ddm_distr(n, x0, t0, a, s, z, dt)
            DOCUMENTATION NEEDED
            from Michael Shvartsman
        - ddm_analytic(bais, T0, drift_rate, noise, threshold)
            DOCUMENTATION NEEDED
            from Michael Shvartsman

    """

    functionType = "DDM"

    classPreferenceLevel = PreferenceLevel.SUBTYPE
    # These will override those specified in SubtypeDefaultPreferences
    classPreferences = {
        kwPreferenceSetName: 'DDMCustomClassPreferences',
        kpReportOutputPref: PreferenceEntry(True, PreferenceLevel.SUBTYPE)}

    # Assigned in __init__ to match default staring_point
    variableClassDefault = None

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        # Assign internal params here (not accessible to user)
        # User accessible params are assigned in assign_defaults_to_paramClassDefaults (in __init__)
        kwOutputStates:[kwDDM_DecisionVariable,      # Full set specified to include Navarro and Fuss outputs
                        kwDDM_Error_Rate,            # If Bogacz is implemented, last four are deleted
                        kwDDM_Probability_upperBound, # Probability of hitting upper bound
                        kwDDM_Probability_lowerBound, # Probability of hitting lower bound
                        kwDDM_RT_Mean],               #    in instantiate_function (see below)
                        # kwDDM_RT_Correct_Mean,
                        # kwDDM_RT_Correct_Variance,
                        # kwDDM_Total_Allocation,
                        # kwDDM_Total_Cost],
        # MONITORED_OUTPUT_STATES:[kwDDM_Error_Rate,(kwDDM_RT_Mean, -1, 1)]
    })

    # Set default input_value to default bias for DDM
    paramNames = paramClassDefaults.keys()

    def __init__(self,
                 default_input_value=NotImplemented,
                 function=BogaczEtAl(drift_rate=1.0,
                                     starting_point=0.0,
                                     threshold=1.0,
                                     noise=0.5,
                                     T0=.200),
                 name=NotImplemented,
                 prefs=NotImplemented,
                 params=None,
                 context=NotImplemented):
        """Assign type-level preferences, default input value (DDM_Defaults.starting_point) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(function=function,
                                                 params=params)

        self.variableClassDefault = self.paramClassDefaults[FUNCTION_PARAMS][kwDDM_StartingPoint]

        if default_input_value is NotImplemented:
            default_input_value = params[FUNCTION_PARAMS][kwDDM_StartingPoint]

        super(DDM, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context,
                                  context=self)

    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=NotImplemented):
        """Execute DDM function (currently only trial-level, analytic solution)

        Execute DDM and estimate outcome or calculate trajectory of decision variable
        Currently implements only trial-level DDM (analytic solution) and returns:
            - stochastically estimated decion outcome (convert mean ER into value between 1 and -1)
            - mean ER
            - mean DT
            - mean ER and DT variabilty (kwNavarroAndFuss ony)
        Return current decision variable (self.outputState.value) and other output values (self.outputStates[].value

        Arguments:

        # CONFIRM:
        variable (float): set to self.value (= self.inputValue)
        - params (dict):  runtime_params passed from Mechanism, used as one-time value for current execution:
            + kwDDM_DriftRate (float)
            + kwDDM_Threshold (float)
            + kwDDM_Bias (float)
            + kwDDM_T0 (float)
            + kwDDM_Noise (float)
        - time_scale (TimeScale): determines "temporal granularity" with which mechanism is executed
        - context (str)

        Returns the following values in self.value (2D np.array) and in
            the value of the corresponding outputState in the self.outputStates dict:
            - decision variable (float)
            - mean error rate (float)
            - mean RT (float)
            - correct mean RT (float) - Navarro and Fuss only
            - correct mean ER (float) - Navarro and Fuss only

        :param self:
        :param variable (float)
        :param params: (dict)
        :param time_scale: (TimeScale)
        :param context: (str)
        :rtype self.outputState.value: (number)
        """

        #region EXECUTE INTEGRATOR SOLUTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for DDM")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset output
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Implement terminate() below
        #endregion

        #region EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # Get length of output from kwOutputStates
            # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            #       is run (to evaluate output) before outputStates have been instantiated
            output = [None] * len(self.paramsCurrent[kwOutputStates])
            #     output = np.array([[None]]*len(self.paramsCurrent[kwOutputStates]))
            # Assign output mappings:
            self.outputStateValueMapping = {}
            self.outputStateValueMapping[kwDDM_DecisionVariable] = DDM_Output.DECISION_VARIABLE.value
            self.outputStateValueMapping[kwDDM_RT_Mean] = DDM_Output.RT_MEAN.value
            self.outputStateValueMapping[kwDDM_Error_Rate] = DDM_Output.ER_MEAN.value
            self.outputStateValueMapping[kwDDM_Probability_upperBound] = DDM_Output.P_UPPER_MEAN.value
            self.outputStateValueMapping[kwDDM_Probability_lowerBound] = DDM_Output.P_LOWER_MEAN.value

            # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
            drift_rate = float((self.inputState.value * self.parameterStates[kwDDM_DriftRate].value))
            starting_point = float(self.parameterStates[kwDDM_StartingPoint].value)
            threshold = float(self.parameterStates[kwDDM_Threshold].value)
            noise = float(self.parameterStates[kwDDM_Noise].value)
            T0 = float(self.parameterStates[kwDDM_T0].value)

            result = self.function(params={kwDDM_DriftRate:drift_rate,
                                           kwDDM_StartingPoint:starting_point,
                                           kwDDM_Threshold:threshold,
                                           kwDDM_Noise:noise,
                                           kwDDM_T0:T0})

            output[DDM_Output.RT_MEAN.value], output[DDM_Output.ER_MEAN.value] = result
            output[DDM_Output.P_UPPER_MEAN.value] = 1 - output[DDM_Output.ER_MEAN.value]
            output[DDM_Output.P_LOWER_MEAN.value] = output[DDM_Output.ER_MEAN.value]

            # #region Navarro and Fuss solution:
            # elif self.paramsCurrent[kwDDM_AnalyticSolution] is kwDDM_NavarroAndFuss:
            #     print("\nimporting matlab...")
            #     import matlab.engine
            #     eng1 = matlab.engine.start_matlab('-nojvm')
            #     print("matlab imported\n")
            #     results = eng1.ddmSim(drift_rate, bias, threshold, noise, T0, 1, nargout=5)
            #     output[DDM_Output.RT_MEAN.value] = results[NF_Results.MEAN_DT.value]
            #     output[DDM_Output.ER_MEAN.value] = 1-results[NF_Results.MEAN_ER.value]
            #     output[DDM_Output.P_UPPER_MEAN.value] = results[NF_Results.MEAN_ER.value]
            #     output[DDM_Output.P_LOWER_MEAN.value] = 1 - results[NF_Results.MEAN_ER.value]
            #     output[DDM_Output.RT_CORRECT_MEAN.value] = results[NF_Results.MEAN_CORRECT_RT.value]
            #     output[DDM_Output.RT_CORRECT_VARIANCE.value] = results[NF_Results.MEAN_CORRECT_VARIANCE.value]
            #     # CORRECT_RT_SKEW = results[DDMResults.MEAN_CORRECT_SKEW_RT.value]
            #
            #     self.outputStateValueMapping[kwDDM_Probability_upperBound] = DDM_Output.P_UPPER_MEAN.value
            #     self.outputStateValueMapping[kwDDM_Probability_lowerBound] = DDM_Output.P_LOWER_MEAN.value
            #     self.outputStateValueMapping[kwDDM_RT_Correct_Mean] = DDM_Output.RT_CORRECT_MEAN.value
            #     self.outputStateValueMapping[kwDDM_RT_Correct_Variance] = DDM_Output.RT_CORRECT_VARIANCE.value
            #
            # #endregion

            # **********************************************************************************************************
            # **********************************************************************************************************


            #region Convert ER to decision variable:
            if random() < output[DDM_Output.ER_MEAN.value]:
                output[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(-1 * threshold)
            else:
                output[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(threshold)
            #endregion

            #region Print results
            # if (self.prefs.reportOutputPref and kwFunctionInit not in context):
            if self.prefs.reportOutputPref and kwExecuting in context:
                print ("\n{0} execute method:\n- input: {1}\n- params:".
                       format(self.name, self.inputState.value.__str__().strip("[]")))
                print ("    drift:", drift_rate,
                       "\n    starting point:", starting_point,
                       "\n    thresh:", threshold,
                       "\n    T0:", T0,
                       "\n    noise:", noise,
                       "\n- output:",
                       # "\n    mean ER: {:.3f}".format(output[DDM_Output.ER_MEAN.value]),
                       # "\n    mean RT: {:.3f}".format(output[DDM_Output.RT_MEAN.value]))
                       "\n    mean P(upper bound): {:.3f}".format(float(output[DDM_Output.P_UPPER_MEAN.value])),
                       "\n    mean P(lower bound): {:.3f}".format(float(output[DDM_Output.P_LOWER_MEAN.value])),
                       "\n    mean ER: {:.3f}".format(float(output[DDM_Output.ER_MEAN.value])),
                       "\n    mean RT: {:.3f}".format(float(output[DDM_Output.RT_MEAN.value])))
                # if self.paramsCurrent[kwDDM_AnalyticSolution] is kwDDM_NavarroAndFuss:
                #     print(
                #         "Correct RT Mean: {:.3f}".format(output[DDM_Output.RT_CORRECT_MEAN.value]),
                #         "\nCorrect RT Variance: {:.3f}".format(output[DDM_Output.RT_CORRECT_VARIANCE.value]))
                #         # "\nMean Correct RT Skewy:", CORRECT_RT_SKEW)
                print ("Output: ", output[DDM_Output.DECISION_VARIABLE.value].__str__().strip("[]"))
            #endregion

            return output
        #endregion

        else:
            raise MechanismError("time_scale not specified for DDM")

    def ou_update(self, particle, drift, noise, time_step_size, decay):
        ''' Single update for OU (special case l=0 is DDM)'''
        return particle + time_step_size * (decay * particle + drift) + random.normal(0, noise) * sqrt(time_step_size)


    def ddm_update(self, particle, a, s, dt):
        return self.ou_update(particle, a, s, dt, decay=0)


    def ddm_rt(self, x0, t0, a, s, z, dt):
        samps = 0
        particle = x0
        while abs(particle) < z:
            samps = samps + 1
            particle = self.ou_update(particle, a, s, dt, decay=0)
        # return -rt for errors as per convention
        return (samps * dt + t0) if particle > 0 else -(samps * dt + t0)

    def ddm_distr(self, n, x0, t0, a, s, z, dt):
        return np.fromiter((self.ddm_rt(x0, t0, a, s, z, dt) for i in range(n)), dtype='float64')


    def terminate_function(self, context=NotImplemented):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for DDM


