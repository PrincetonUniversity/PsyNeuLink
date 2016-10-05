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
DECISION_VARIABLE = "DecisionVariable"
ERROR_RATE = "Error_Rate"
PROBABILITY_UPPER_BOUND = "Probability_upperBound"
PROBABILITY_LOWER_BOUND = "Probability_lowerBound"
RT_MEAN = "RT_Mean"
RT_CORRECT_MEAN = "RT_Correct_Mean"
RT_CORRECT_VARIANCE = "RT_Correct_Variance"
# TOTAL_ALLOCATION = "Total_Allocation"
# TOTAL_COST = "Total_Cost"


# Indices for results used in return value tuple; auto-numbered to insure sequentiality
class DDM_Output(AutoNumber):
    DECISION_VARIABLE = ()
    RT_MEAN = ()
    ER_MEAN = ()
    P_UPPER_MEAN = ()
    P_LOWER_MEAN = ()
    RT_CORRECT_MEAN = ()
    RT_CORRECT_VARIANCE = ()
    # TOTAL_COST = ()
    # TOTAL_ALLOCATION = ()
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
            - Bogacz et al. (2006) analytic solution (see kwBogaczEtAl option below):
                generates error rate (ER) and decision time (DT);
                ER is used to stochastically generate a decision outcome (+ or - valued) on every run
            - Navarro and Fuss (2009) analytic solution (see kwNavarrosAndFuss:
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
            + FUNCTION (Integrator):
                specifies analytic solution of the DDM to use;
                Must be either BogaczEtAl or NavarroAndFuss
                + BogaczEtAl: generates mean reaction time (RT) and mean error rate (ER) as described in:
                    Bogacz, R., Brown, E., Moehlis, J., Holmes, P., & Cohen, J. D. (2006). The physics of optimal
                        decision making: a formal analysis of models of performance in two-alternative forced-choice
                        tasks.  Psychological review, 113(4), 700.
                    Notes:
                    * requires that DDM.execute be called with time_scale = TimeScale.TRIAL
                + NavarrosAndFuss:  gives mean RT and ER as well as distributions as described in:
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
                + DRIFT_RATE (float):
                    specifies internal ("attentional") component of the drift rate
                    that is added to the input (self.variable) on every call to DDM.execute()
                + STARTING_POINT (float):
                    specifies intitial value of decision variable, converted to "bias" term in DDM
                + THRESHOLD (float):
                    specifies stopping value of decision variable for integration process
                + NOISE (float):
                    specifies internal noise term for integration process
                + NON_DECISION_TIME (float):
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
                                      kwDDM_AnalyticSolution: kwBogaczEtAl,
                                      FUNCTION_PARAMS: {DRIFT_RATE:<>
                                                              STARTING_POINT:<>
                                                              THRESHOLD:<>
                                                              NOISE:<>
                                                              NON_DECISION_TIME:<>},
                                      kwOutputStates: [DECISION_VARIABLE,
                                                       ERROR_RATE,
                                                       PROBABILITY_UPPER_BOUND,
                                                       PROBABILITY_LOWER_BOUND,
                                                       RT_MEAN,
                                                       RT_CORRECT_MEAN,
                                                       RT_CORRECT_VARIANCE,
                                                       TOTAL_ALLOCATION,
                                                       TOTAL_COST],
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
        kpReportOutputPref: PreferenceEntry(False, PreferenceLevel.INSTANCE)}

    # Assigned in __init__ to match default staring_point
    variableClassDefault = None

    paramClassDefaults = Mechanism_Base.paramClassDefaults.copy()
    paramClassDefaults.update({
        kwTimeScale: TimeScale.TRIAL,
        # Assign internal params here (not accessible to user)
        # User accessible params are assigned in assign_defaults_to_paramClassDefaults (in __init__)
        kwOutputStates:[DECISION_VARIABLE,      # Full set specified to include Navarro and Fuss outputs
                        ERROR_RATE,            # If Bogacz is implemented, last four are deleted
                        PROBABILITY_UPPER_BOUND, # Probability of hitting upper bound
                        PROBABILITY_LOWER_BOUND, # Probability of hitting lower bound
                        RT_MEAN,               #    in instantiate_function (see below)
                        RT_CORRECT_MEAN,
                        RT_CORRECT_VARIANCE]
                        # TOTAL_ALLOCATION,
                        # TOTAL_COST]
        # MONITORED_OUTPUT_STATES:[ERROR_RATE,(RT_MEAN, -1, 1)]
    })

    # Set default input_value to default bias for DDM
    paramNames = paramClassDefaults.keys()


    @tc.typecheck
    def __init__(self,
                 default_input_value=NotImplemented,
                 # function:tc.enum(type(BogaczEtAl), type(NavarroAndFuss))=BogaczEtAl(drift_rate=1.0,
                 function=BogaczEtAl(drift_rate=1.0,
                                     starting_point=0.0,
                                     threshold=1.0,
                                     noise=0.5,
                                     T0=.200),
                 name=None,
                 params=None,
                 # prefs:tc.optional(FunctionPreferenceSet)=None,
                 prefs:is_pref_set=None,
                 context=None):
        """Assign type-level preferences, default input value (DDM_Defaults.starting_point) and call super.__init__

        :param default_input_value: (value)
        :param params: (dict)
        :param name: (str)
        :param prefs: (PreferenceSet)
        """

        # Assign args to params and functionParams dicts (kwConstants must == arg names)
        params = self.assign_args_to_param_dicts(function=function,
                                                 params=params)

        self.variableClassDefault = self.paramClassDefaults[FUNCTION_PARAMS][STARTING_POINT]

        if default_input_value is NotImplemented:
            default_input_value = params[FUNCTION_PARAMS][STARTING_POINT]

        super(DDM, self).__init__(variable=default_input_value,
                                  params=params,
                                  name=name,
                                  prefs=prefs,
                                  # context=context,
                                  context=self)

    def validate_params(self, request_set, target_set=NotImplemented, context=None):

        super().validate_params(request_set=request_set, target_set=target_set, context=context)

        functions = {BogaczEtAl, NavarroAndFuss}
        if not target_set[FUNCTION] in functions:
            function_names = list(function.functionName for function in functions)
            raise DDMError("{} param of {} must be one of the following functions: {}".
                           format(self.name, function_names))

    # def instantiate_function(self, context=NotImplemented):
    def instantiate_attributes_before_function(self, context=None):
        """Delete params not in use, call super.instantiate_execute_method
        :param context:
        :return:
        """

        # Assign output mappings:
        self.outputStateValueMapping = {}
        self.outputStateValueMapping[DECISION_VARIABLE] = DDM_Output.DECISION_VARIABLE.value
        self.outputStateValueMapping[RT_MEAN] = DDM_Output.RT_MEAN.value
        self.outputStateValueMapping[ERROR_RATE] = DDM_Output.ER_MEAN.value
        self.outputStateValueMapping[PROBABILITY_UPPER_BOUND] = DDM_Output.P_UPPER_MEAN.value
        self.outputStateValueMapping[PROBABILITY_LOWER_BOUND] = DDM_Output.P_LOWER_MEAN.value

        # If not using Navarro and Fuss, get rid of extra params:
        if self.function is BogaczEtAl:
            outputStates = self.params[kwOutputStates]
            try:
                del outputStates[outputStates.index(RT_CORRECT_MEAN)]
                del outputStates[outputStates.index(RT_CORRECT_VARIANCE)]
                # del outputStates[outputStates.index(TOTAL_ALLOCATION)]
                # del outputStates[outputStates.index(TOTAL_COST)]
            except ValueError:
                pass
        else:
            self.outputStateValueMapping[RT_CORRECT_MEAN] = DDM_Output.RT_CORRECT_MEAN.value
            self.outputStateValueMapping[RT_CORRECT_VARIANCE] = DDM_Output.RT_CORRECT_VARIANCE.value
            # self.outputStateValueMapping[TOTAL_ALLOCATION] = DDM_Output.TOTAL_ALLOCATION.value
            # self.outputStateValueMapping[TOTAL_COST] = DDM_Output.TOTAL_COST.value

        super().instantiate_attributes_before_function(context=context)

    def __execute__(self,
                variable=NotImplemented,
                params=NotImplemented,
                time_scale = TimeScale.TRIAL,
                context=None):
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
            + DRIFT_RATE (float)
            + THRESHOLD (float)
            + kwDDM_Bias (float)
            + NON_DECISION_TIME (float)
            + NOISE (float)
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

        # EXECUTE INTEGRATOR SOLUTION (REAL_TIME TIME SCALE) -----------------------------------------------------
        if time_scale == TimeScale.REAL_TIME:
            raise MechanismError("REAL_TIME mode not yet implemented for DDM")
            # IMPLEMENTATION NOTES:
            # Implement with calls to a step_function, that does not reset self.outputValue
            # Should be sure that initial value of self.outputState.value = self.parameterStates[BIAS]
            # Implement terminate() below

        # EXECUTE ANALYTIC SOLUTION (TRIAL TIME SCALE) -----------------------------------------------------------
        elif time_scale == TimeScale.TRIAL:

            # # Get length of self.outputValue from kwOutputStates
            # # Note: use paramsCurrent here (instead of outputStates), as during initialization the execute method
            # #       is run (to evaluate self.outputValue) before outputStates have been instantiated
            # self.outputValue = [None] * len(self.paramsCurrent[kwOutputStates])

            # # TEST PRINT:
            # print ("\nDDM RUN")
            # print ("stimulus: {}".format(self.inputState.value))
            # print ("control signal: {}\n".format(self.parameterStates[DRIFT_RATE].value))

            # - convolve inputState.value (signal) w/ driftRate param value (attentional contribution to the process)
            drift_rate = float((self.inputState.value * self.parameterStates[DRIFT_RATE].value))
            starting_point = float(self.parameterStates[STARTING_POINT].value)
            threshold = float(self.parameterStates[THRESHOLD].value)
            noise = float(self.parameterStates[NOISE].value)
            T0 = float(self.parameterStates[NON_DECISION_TIME].value)

            result = self.function(params={DRIFT_RATE:drift_rate,
                                           STARTING_POINT:starting_point,
                                           THRESHOLD:threshold,
                                           NOISE:noise,
                                           NON_DECISION_TIME:T0})

            # Assign outputValue

            if isinstance(self.function.__self__, BogaczEtAl):
                self.outputValue[DDM_Output.RT_MEAN.value], self.outputValue[DDM_Output.ER_MEAN.value] = result
                self.outputValue[DDM_Output.P_UPPER_MEAN.value] = 1 - self.outputValue[DDM_Output.ER_MEAN.value]
                self.outputValue[DDM_Output.P_LOWER_MEAN.value] = self.outputValue[DDM_Output.ER_MEAN.value]

            elif isinstance(self.function.__self__, NavarroAndFuss):
                self.outputValue[DDM_Output.RT_MEAN.value] = result[NF_Results.MEAN_DT.value]
                self.outputValue[DDM_Output.ER_MEAN.value] = 1-result[NF_Results.MEAN_ER.value]
                self.outputValue[DDM_Output.P_UPPER_MEAN.value] = result[NF_Results.MEAN_ER.value]
                self.outputValue[DDM_Output.P_LOWER_MEAN.value] = 1 - result[NF_Results.MEAN_ER.value]
                self.outputValue[DDM_Output.RT_CORRECT_MEAN.value] = result[NF_Results.MEAN_CORRECT_RT.value]
                self.outputValue[DDM_Output.RT_CORRECT_VARIANCE.value] = result[NF_Results.MEAN_CORRECT_VARIANCE.value]
                # CORRECT_RT_SKEW = results[DDMResults.MEAN_CORRECT_SKEW_RT.value]

            # Convert ER to decision variable:
            if random() < self.outputValue[DDM_Output.ER_MEAN.value]:
                self.outputValue[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(-1 * threshold)
            else:
                # # MODIFIED 10/5/16 OLD:
                # self.outputValue[DDM_Output.DECISION_VARIABLE.value] = np.atleast_1d(threshold)
                # MODIFIED 10/5/16 NEW:
                self.outputValue[DDM_Output.DECISION_VARIABLE.value] = threshold
                # MODIFIED 10/5/16 END

            return self.outputValue

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


    def terminate_function(self, context=None):
        """Terminate the process

        called by process.terminate() - MUST BE OVERRIDDEN BY SUBCLASS IMPLEMENTATION
        returns output

        :rtype CurrentStateTuple(state, confidence, duration, controlModulatedParamValues)
        """
        # IMPLEMENTATION NOTE:  TBI when time_step is implemented for DDM