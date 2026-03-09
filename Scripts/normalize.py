
class Normalize(DeterministicTransferFunction):
    """
    Normalize(                      \
         default_variable,          \
         eps=1e-12                  \
         per_item=True              \
         params=None,               \
         owner=None,                \
         name=None,                 \
         prefs=None                 \
         )

    .. _Normalize:

    Returns the normalized value of `variable <Normalize.variable>`.

    `function <Normalize._function>` returns the L2-normalized value of `variable <Normalize.variable>`:

    .. math::
        f(x) = \\frac{x}{\\max(\\lVert x \\rVert_2, \\epsilon)}

    where :math:`\\lVert x \\rVert_2` is the Euclidean norm of :math:`x`, and :math:`\\epsilon` is a small positive constant used
    for numerical stability.

    .. note::
        If :math:`\\lVert x \\rVert_2 \\le \\epsilon`, the denominator is clamped to :math:`\\epsilon`.

    .. _Normalize_Derivative:

    *Derivative*

    `derivative <Normalize.derivative>` returns the Jacobian of the Normalize function.

    .. math::
        J_{ij} = \\frac{\\partial f_i}{\\partial x_j}

    When :math:`\\lVert x \\rVert_2 > \\epsilon`:

    .. math::
        J = \\frac{1}{\\lVert x \\rVert_2} I - \\frac{x x^T}{\\lVert x \\rVert_2^3}

    When :math:`\\lVert x \\rVert_2 \\le \\epsilon`:

    .. math::
        J = \\frac{1}{\\epsilon} I

    Arguments
    ---------

    default_variable : 1d array : default class_defaults.variable
        specifies a template for the value to be transformed.

    eps: float : default 1e-12
        a small positive constant used for numerical stability when the norm of the input is close to zero
        the denominator of the normalization is clamped to this value to prevent division by zero.

    per_item : boolean : default True
        for 2d variables, determines whether Normalize is applied to the entire variable
        (*per_item* = False), or applied to each item in the variable separately
        (*per_item* = True).

    params : Dict[param keyword: param value] : default None
        a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
        function.  Values specified for parameters in the dictionary override any assigned to those parameters in
        arguments of the constructor.

    owner : Component
        `component <Component>` to which to assign the Function.

    name : str : default see `name <Function.name>`
        specifies the name of the Function.

    prefs : PreferenceSet or specification dict : default Function.classPreferences
        specifies the `PreferenceSet` for the Function (see `prefs <Function_Base.prefs>` for details).

    Attributes
    ----------

    variable : 1d array
        contains value to be transformed.

    eps: float : default 1e-12
        a small positive constant used for numerical stability when the norm of the input is close to zero
        the denominator of the normalization is clamped to this value to prevent division by zero.

    per_item : boolean : default True
        for 2d variables, determines whether Normalize is applied to the entire variable
        (*per_item* = False), or applied to each item in the variable separately
        (*per_item* = True).

    range : (None, None)

    owner : Component
        `component <Component>` to which the Function has been assigned.

    name : str
        the name of the Function; if it is not specified in the **name** argument of the constructor, a default is
        assigned by FunctionRegistry (see `Registry_Naming` for conventions used for default and duplicate names).

    prefs : PreferenceSet or specification dict : Function.classPreferences
        the `PreferenceSet` for function; if it is not specified in the **prefs** argument of the Function's
        constructor, a default is assigned using `classPreferences` defined in __init__.py (see `Preferences`
        for details).
    """

    componentName = NORMALIZE_FUNCTION
    default_range = (None, None)

    class Parameters(TransferFunction.Parameters):
        """
            Attributes
            ----------

                variable
                    see `variable <Normalize.variable>`

                    :default value: numpy.array(0.)
                    :type: ``numpy.ndarray``
                    :read only: True

                eps
                    see `eps <Normalize.eps>`

                    :default value: 1e-12
                    :type: ``float``

                per_item
                    see `per_item <Normalize.per_item>`

                    :default value: True
                    :type: ``bool``


                range
                    see `range <Normalize.range>`

                    :default value: (None, None)
                    :type: <class 'tuple'>

        """
        variable = Parameter(np.array([[0.0]]), read_only=True, pnl_internal=True,
                             constructor_argument='default_variable')
        eps = Parameter(1e-12, modulable=False)
        per_item = Parameter(True, modulable=False)


    @check_user_specified
    @beartype
    def __init__(self,
                 default_variable=None,
                 eps = None,
                 per_item = None,
                 params: Optional[Mapping] = None,
                 owner=None,
                 prefs: Optional[ValidPrefSet] = None):

        super().__init__(
            default_variable=default_variable,
            eps=eps,
            per_item=per_item,
            params=params,
            owner=owner,
            prefs=prefs,
        )

    def _validate_variable(self, variable, context=None):
        if variable is None:
            try:
                return self.defaults.variable
            except AttributeError:
                return self.class_defaults.variable
        return np.asarray(variable)


    def _normalize(self, input_value, eps):
        nrm = np.linalg.norm(input_value)
        denominator = max(nrm, eps)
        return input_value / denominator

    def _function(self,
                  variable=None,
                  context=None,
                  params=None):
        """
        Arguments
        ---------

        variable : 1d array : default class_defaults.variable
           an array to be transformed.

        params : Dict[param keyword: param value] : default None
            a `parameter dictionary <ParameterPort_Specification>` that specifies the parameters for the
            function.  Values specified for parameters in the dictionary override any assigned to those parameters in
            arguments of the constructor.

        Returns
        -------
        Normalized variable : number or array
        """
        eps = self._get_current_parameter_value('eps', context)
        per_item = self._get_current_parameter_value('per_item', context)

        if per_item and len(np.shape(variable)) > 1:
            output = []
            for item in variable:
                output.append(self._normalize(item, eps))
            output = convert_all_elements_to_np_array(output)
        else:
            output = self._normalize(variable, eps)

        return self.convert_output_type(output)

    @handle_external_context()
    def derivative(self, input=None, output=None, context=None):

        if input is None:
            raise FunctionError(
                f"Derivative of Normalize for '{self.owner.name}' requires 'input'."
            )

        eps = self._get_current_parameter_value('eps', context)
        per_item = self._get_current_parameter_value('per_item', context)

        if per_item and len(np.shape(input)) > 1:
            result = []
            for item in input:
                x = np.asarray(item, dtype=float)
                nrm = np.linalg.norm(x)
                size = x.shape[0]
                I = np.eye(size)

                if nrm <= eps:
                    result.append(I / eps)
                else:
                    result.append(I / nrm - np.outer(x, x) / (nrm ** 3))

            return result

        x = np.asarray(input, dtype=float)
        nrm = np.linalg.norm(x)

        size = x.shape[0]
        I = np.eye(size)

        if nrm <= eps:
            return I / eps

        return I / nrm - np.outer(x, x) / (nrm ** 3)

    def __gen_llvm_sq_sum(self, builder, index, ctx, vi, sq_sum_ptr):
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])

        orig_val = builder.load(ptri)
        sq_val = builder.fmul(orig_val, orig_val)

        sq_sum = builder.load(sq_sum_ptr)
        new_sq_sum = builder.fadd(sq_sum, sq_val)
        builder.store(new_sq_sum, sq_sum_ptr)

    def __gen_llvm_norm_div(self, builder, index, ctx, vi, vo, denom):
        ptro = builder.gep(vo, [ctx.int32_ty(0), index])
        ptri = builder.gep(vi, [ctx.int32_ty(0), index])

        val = builder.load(ptri)
        val = builder.fdiv(val, denom)

        builder.store(val, ptro)

    def __gen_llvm_apply(self, ctx, builder, params, state, arg_in, arg_out, tags: frozenset):
        sq_sum_ptr = builder.alloca(ctx.float_ty)
        builder.store(sq_sum_ptr.type.pointee(0), sq_sum_ptr)

        eps_ptr = ctx.get_param_or_state_ptr(builder, self, 'eps', param_struct_ptr=params)
        eps = pnlvm.helpers.load_extract_scalar_array_one(builder, eps_ptr)

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "sq_sum") as args:
            self.__gen_llvm_sq_sum(*args, ctx=ctx, vi=arg_in, sq_sum_ptr=sq_sum_ptr)

        sq_sum = builder.load(sq_sum_ptr)

        sqrt_f = ctx.get_builtin("sqrt", [ctx.float_ty])
        nrm = builder.call(sqrt_f, [sq_sum])

        # denom = max(nrm, eps)
        use_nrm = builder.fcmp_ordered(">=", nrm, eps)
        denom = builder.select(use_nrm, nrm, eps)

        with pnlvm.helpers.array_ptr_loop(builder, arg_in, "norm_div") as args:
            self.__gen_llvm_norm_div(*args, ctx=ctx, vi=arg_in, vo=arg_out, denom=denom)

        return builder

    def _gen_llvm_function_derivative_body(self, ctx, builder, params, state, arg_in, arg_out, *, tags: frozenset):
        assert "derivative" in tags or "derivative_out" in tags
        assert arg_in.type == arg_out.type
        forward_tags = tags.difference({"derivative", "derivative_out"})

        # Normalize derivative is computed from the forward output.
        if "derivative_out" in tags:
            all_out = arg_in
        else:
            all_out = builder.alloca(arg_out.type.pointee)
            builder = self._gen_llvm_function_body(
                ctx, builder, params, state, arg_in, all_out, tags=forward_tags
            )

        if self.parameters.per_item.get():
            assert isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)
            assert isinstance(arg_out.type.pointee.element, pnlvm.ir.ArrayType)
            for i in range(arg_in.type.pointee.count):
                inner_all_out = builder.gep(all_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                inner_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                builder = self.__gen_llvm_apply_derivative(
                    ctx, builder, params, state, inner_all_out, inner_out, tags=tags
                )
            return builder
        else:
            return self.__gen_llvm_apply_derivative(ctx, builder, params, state, all_out, arg_out, tags=tags)

    def __gen_llvm_apply_derivative(self, ctx, builder, params, state, all_out, arg_out, *, tags: frozenset):
        # NOTE:
        # Normalize has a full Jacobian, since each output element depends on every input element.
        # However, the current LLVM derivative interface expects arg_out to have the same shape as arg_in,
        # so for now we return only the diagonal of the Jacobian as a vector-shaped approximation.
        # This matches the existing derivative plumbing, but is not the full mathematical derivative.

        eps_ptr = ctx.get_param_or_state_ptr(builder, self, 'eps', param_struct_ptr=params)
        eps = pnlvm.helpers.load_extract_scalar_array_one(builder, eps_ptr)

        out_sq_sum_ptr = builder.alloca(ctx.float_ty)
        builder.store(out_sq_sum_ptr.type.pointee(0), out_sq_sum_ptr)

        # Compute ||all_out||^2
        with pnlvm.helpers.array_ptr_loop(builder, all_out, id="out_sq_sum") as (b, idx):
            val_ptr = b.gep(all_out, [ctx.int32_ty(0), idx])
            val = b.load(val_ptr)
            val_sq = b.fmul(val, val)

            cur_sum = b.load(out_sq_sum_ptr)
            new_sum = b.fadd(cur_sum, val_sq)
            b.store(new_sum, out_sq_sum_ptr)

        sqrt_f = ctx.get_builtin("sqrt", [ctx.float_ty])
        out_sq_sum = builder.load(out_sq_sum_ptr)
        out_norm = builder.call(sqrt_f, [out_sq_sum])

        one = ctx.float_ty(1)
        inv_eps = builder.fdiv(one, eps)

        # If ||all_out|| < 1, then we are in the clamped branch: y = x / eps
        is_clamped = builder.fcmp_ordered("<", out_norm, one)

        with pnlvm.helpers.array_ptr_loop(builder, all_out, id="derivative") as (b, idx):
            y_ptr = b.gep(all_out, [ctx.int32_ty(0), idx])
            y = b.load(y_ptr)

            # unclamped diagonal derivative:
            # J_ii = (1 - y_i^2) / ||x||
            # and since ||y|| = 1 in this branch, ||x|| = 1 / ||y_before_clamp_relation|| is not recoverable from y alone.
            # But for normalized output y = x / ||x||, we use the standard diagonal form in terms of y with ||x||=1 scale.
            #
            # For now, keep the simple approximation:
            # J_ii ≈ 1 - y_i^2
            y_sq = b.fmul(y, y)
            unclamped = b.fsub(one, y_sq)

            val = b.select(is_clamped, inv_eps, unclamped)

            out_ptr = b.gep(arg_out, [ctx.int32_ty(0), idx])
            b.store(val, out_ptr)

        return builder

    def _gen_llvm_function_body(self, ctx, builder, params, state, arg_in, arg_out, *,
                                tags: frozenset):
        if "derivative" in tags or "derivative_out" in tags:
            return self._gen_llvm_function_derivative_body(ctx, builder, params, state, arg_in, arg_out, tags=tags)

        if self.parameters.per_item.get():
            assert isinstance(arg_in.type.pointee.element, pnlvm.ir.ArrayType)
            assert isinstance(arg_out.type.pointee.element, pnlvm.ir.ArrayType)
            for i in range(arg_in.type.pointee.count):
                inner_in = builder.gep(arg_in, [ctx.int32_ty(0), ctx.int32_ty(i)])
                inner_out = builder.gep(arg_out, [ctx.int32_ty(0), ctx.int32_ty(i)])
                builder = self.__gen_llvm_apply(ctx, builder, params, state, inner_in, inner_out, tags=tags)
            return builder
        else:
            return self.__gen_llvm_apply(ctx, builder, params, state, arg_in, arg_out, tags=tags)

    def _gen_pytorch_fct(self, device, context=None):
        eps = self._get_pytorch_fct_param_value('eps', device, context)
        per_item = self._get_pytorch_fct_param_value('per_item', device, context)

        def pytorch_normalize(x: torch.Tensor) -> torch.Tensor:
            if per_item and x.ndim > 1:
                norm = torch.linalg.norm(x, ord=2, dim=-1, keepdim=True)
            else:
                norm = torch.linalg.norm(x, ord=2)
            denom = torch.clamp(norm, min=eps)
            return x / denom

        return pytorch_normalize

