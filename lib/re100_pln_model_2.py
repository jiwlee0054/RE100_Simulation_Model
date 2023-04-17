import pyomo.environ as pyo

from lib.parameters import ParameterPyomoForm, ProbOptions, SetTime


class PortfolioModel:
    def __init__(self, options: ProbOptions, input_parameters_pyomo: ParameterPyomoForm, ST):
        self.model = pyo.ConcreteModel()
        self.model = self.define_sets(input_parameters_pyomo, ST)
        self.model = self.define_parameters(input_parameters_pyomo)
        self.model = self.define_variables()
        self.model = self.define_objective_function()
        self.model = self.define_constraints(options, ST)
        self.print_message()

    def define_sets(self, input_parameters_pyomo: ParameterPyomoForm, ST: SetTime):
        self.model.set_year = pyo.Set(initialize=ST.set_year)
        self.model.set_day = pyo.Set(initialize=ST.set_day)
        self.model.set_hour = pyo.Set(initialize=ST.set_hour)
        self.model.set_storage = pyo.Set(initialize=input_parameters_pyomo.set_storage)
        self.model.set_sg_generator = pyo.Set(initialize=input_parameters_pyomo.set_sg_generator)
        self.model.set_ppa_generator = pyo.Set(initialize=input_parameters_pyomo.set_ppa_generator)
        self.model.set_generator = pyo.Set(initialize=input_parameters_pyomo.set_generator)
        self.model.set_ppa_item = pyo.Set(initialize=input_parameters_pyomo.set_ppa_item)
        self.model.set_trf_item = pyo.Set(initialize=input_parameters_pyomo.set_trf_item)
        self.model.set_tariff = pyo.Set(initialize=input_parameters_pyomo.set_tariff)
        self.model.set_fund = pyo.Set(initialize=input_parameters_pyomo.set_fund)
        self.model.set_tax = pyo.Set(initialize=input_parameters_pyomo.set_tax)
        self.model.set_ueac = pyo.Set(initialize=input_parameters_pyomo.set_ueac)
        # self.model.set_facility = pyo.Set(initialize=input_parameters_pyomo.set_facility)
        self.model.set_historic_year = pyo.Set(initialize=ST.set_historic_year)
        self.model.set_total_year = pyo.Set(initialize=ST.set_historic_year + ST.set_year)
        return self.model

    def define_parameters(self, input_parameters_pyomo: ParameterPyomoForm):
        """
        모델에 반영된 변수
        """
        self.model.para_Discount_y = pyo.Param(self.model.set_year, mutable=True)
        self.model.para_rate_RE100_y = pyo.Param(self.model.set_year, mutable=True)

        self.model.para_DmD_y_d_h = pyo.Param(self.model.set_year, self.model.set_day, self.model.set_hour, mutable=True)

        self.model.para_capa_max_SG_g_y = pyo.Param(self.model.set_sg_generator, self.model.set_year, mutable=True)
        self.model.para_capa_min_SG_g_y = pyo.Param(self.model.set_sg_generator, self.model.set_year, mutable=True)
        self.model.para_price_CPX_g_y = pyo.Param(self.model.set_sg_generator, self.model.set_year, mutable=True)
        self.model.para_price_OPX_g_y = pyo.Param(self.model.set_sg_generator, self.model.set_year, mutable=True)

        self.model.para_capa_max_PPA_g_y = pyo.Param(self.model.set_ppa_generator, self.model.set_year, mutable=True)
        self.model.para_price_PPA_Use_g_y = pyo.Param(self.model.set_ppa_generator, self.model.set_total_year, mutable=True)
        self.model.para_price_PPA_g_i_y = pyo.Param(self.model.set_ppa_generator, self.model.set_ppa_item, self.model.set_year, mutable=True)

        self.model.para_price_TrF_Capa_tf_y = pyo.Param(self.model.set_tariff, self.model.set_year, mutable=True)
        self.model.para_price_TrF_Use_tf_y_d_h = pyo.Param(self.model.set_tariff, self.model.set_year, self.model.set_day, self.model.set_hour, mutable=True)
        self.model.para_price_TrF_i_y = pyo.Param(self.model.set_trf_item, self.model.set_year, mutable=True)

        self.model.para_price_UEAC_i_y = pyo.Param(self.model.set_ueac, self.model.set_year, mutable=True)

        self.model.para_price_inv_ESS_E_capa_s_y = pyo.Param(self.model.set_storage, self.model.set_year, mutable=True)
        self.model.para_price_inv_ESS_P_capa_s_y = pyo.Param(self.model.set_storage, self.model.set_year, mutable=True)
        self.model.para_price_oper_ESS_E_capa_s_y = pyo.Param(self.model.set_storage, self.model.set_year, mutable=True)
        self.model.para_price_oper_ESS_P_capa_s_y = pyo.Param(self.model.set_storage, self.model.set_year, mutable=True)


        self.model.para_rate_Tax_i_y = pyo.Param(self.model.set_tax, self.model.set_year, mutable=True)
        self.model.para_rate_Fund_i_y = pyo.Param(self.model.set_fund, self.model.set_year, mutable=True)

        self.model.para_rate_RVR_SG_g_y = pyo.Param(self.model.set_sg_generator, self.model.set_year, mutable=True)
        self.model.para_rate_RVR_ESS_s_y = pyo.Param(self.model.set_storage, self.model.set_year, mutable=True)


        self.model.para_factor_g_y_d_h = pyo.Param(self.model.set_generator, self.model.set_year, self.model.set_day, self.model.set_hour, mutable=True)

        self.model.para_N_y_d_h = pyo.Param(self.model.set_year, self.model.set_day, self.model.set_hour, mutable=True)

        self.model.para_Big_M = pyo.Param(mutable=True)


        self.model.para_cumul_net_capa_SG_g = pyo.Param(self.model.set_sg_generator,
                                                    mutable=True)
        self.model.para_past_step_capa_SG_g_y = pyo.Param(self.model.set_sg_generator, self.model.set_historic_year,
                                                          mutable=True)
        self.model.para_cumul_net_capa_PPA_g = pyo.Param(self.model.set_ppa_generator,
                                                     mutable=True)
        self.model.para_past_step_capa_PPA_g_y = pyo.Param(self.model.set_ppa_generator, self.model.set_historic_year,
                                                           mutable=True)
        self.model.para_cumul_net_P_capa_ESS_s = pyo.Param(self.model.set_storage,
                                                     mutable=True)
        self.model.para_past_step_P_capa_ESS_s_y = pyo.Param(self.model.set_storage, self.model.set_historic_year,
                                                           mutable=True)

        self.model.para_rate_SOC_upper_ESS_s = pyo.Param(self.model.set_storage, mutable=True)
        self.model.para_rate_SOC_lower_ESS_s = pyo.Param(self.model.set_storage, mutable=True)
        self.model.para_rate_SelfDchg_ESS_s = pyo.Param(self.model.set_storage, mutable=True)
        self.model.para_rate_SOC0_ESS_s = pyo.Param(self.model.set_storage, mutable=True)
        self.model.para_rate_ChrgEff_ESS_s = pyo.Param(self.model.set_storage, mutable=True)
        self.model.para_rate_DchgEff_inv_ESS_s = pyo.Param(self.model.set_storage, mutable=True)

        """
        모델에 미 반영된 변수
        """
        # parameters - storage
        # self.model.para_rate_Crate_s = pyo.Param(self.model.set_storage, mutable=True)

        return self.model

    def define_variables(self):
        """
        모델에 반영된 변수
        """
        self.model.net_capa_SG_g_y = pyo.Var(self.model.set_sg_generator,
                                             self.model.set_year,
                                             bounds=(0, None), initialize=0)
        self.model.step_capa_SG_g_y = pyo.Var(self.model.set_sg_generator,
                                              self.model.set_year,
                                              bounds=(0, None), initialize=0)
        self.model.retire_capa_SG_g_y = pyo.Var(self.model.set_sg_generator,
                                                self.model.set_year,
                                                bounds=(0, None), initialize=0)

        self.model.net_capa_PPA_g_y = pyo.Var(self.model.set_ppa_generator,
                                              self.model.set_year,
                                              bounds=(0, None), initialize=0)
        self.model.step_capa_PPA_g_y = pyo.Var(self.model.set_ppa_generator,
                                               self.model.set_year,
                                               bounds=(0, None), initialize=0)
        self.model.retire_capa_PPA_g_y = pyo.Var(self.model.set_ppa_generator,
                                                 self.model.set_year,
                                                 bounds=(0, None), initialize=0)

        self.model.net_P_capa_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year,
                                        bounds=(0, None), initialize=0)
        self.model.net_E_capa_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year,
                                        bounds=(0, None), initialize=0)
        self.model.step_P_capa_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year,
                                        bounds=(0, None), initialize=0)
        self.model.step_E_capa_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year,
                                        bounds=(0, None), initialize=0)
        self.model.retire_P_capa_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year,
                                        bounds=(0, None), initialize=0)


        self.model.capa_CS_y = pyo.Var(self.model.set_year,
                                       bounds=(0, None), initialize=0)

        self.model.p_SG_g_y_d_h = pyo.Var(self.model.set_sg_generator,
                                          self.model.set_year, self.model.set_day, self.model.set_hour,
                                          bounds=(0, None), initialize=0)

        self.model.p_PPA_g_y_d_h = pyo.Var(self.model.set_ppa_generator,
                                           self.model.set_year, self.model.set_day, self.model.set_hour,
                                           bounds=(0, None), initialize=0)

        self.model.p_TrF_y_d_h = pyo.Var(self.model.set_year, self.model.set_day, self.model.set_hour,
                                         bounds=(0, None), initialize=0)

        self.model.p_UEAC_i_y = pyo.Var(self.model.set_ueac, self.model.set_year,
                                        bounds=(0, None), initialize=0)

        self.model.cost_inv_SG_g_y = pyo.Var(self.model.set_sg_generator,
                                             self.model.set_year,
                                             initialize=0)
        self.model.cost_oper_SG_g_y = pyo.Var(self.model.set_sg_generator,
                                              self.model.set_year,
                                              initialize=0)
        self.model.cost_SG_g = pyo.Var(self.model.set_sg_generator,
                                       initialize=0)

        self.model.cost_PPA_Use_g_y_d_h = pyo.Var(self.model.set_ppa_generator,
                                                  self.model.set_year,
                                                  self.model.set_day,
                                                  self.model.set_hour,
                                                  bounds=(0, None), initialize=0)
        self.model.cost_PPA_g_i_y = pyo.Var(self.model.set_ppa_generator,
                                            self.model.set_ppa_item,
                                            self.model.set_year,
                                            initialize=0)
        self.model.cost_PPA_i = pyo.Var(self.model.set_ppa_item,
                                        initialize=0)

        self.model.cost_inv_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year, initialize=0)
        self.model.cost_oper_ESS_s_y = pyo.Var(self.model.set_storage, self.model.set_year, initialize=0)
        self.model.cost_ESS_s = pyo.Var(self.model.set_storage, initialize=0)

        self.model.cost_TrF_Capa_y = pyo.Var(self.model.set_year,
                                             bounds=(0, None), initialize=0)
        self.model.cost_TrF_Use_y_d_h = pyo.Var(self.model.set_year, self.model.set_day, self.model.set_hour,
                                                bounds=(0, None), initialize=0)
        self.model.cost_TrF_i_y = pyo.Var(self.model.set_trf_item, self.model.set_year,
                                          initialize=0)
        self.model.cost_TrF_i = pyo.Var(self.model.set_trf_item,
                                        initialize=0)

        self.model.cost_RV_SG_g = pyo.Var(self.model.set_sg_generator,
                                          initialize=0)
        self.model.cost_RV_ESS_s = pyo.Var(self.model.set_storage,
                                          initialize=0)

        self.model.cost_Fund_i_y = pyo.Var(self.model.set_fund, self.model.set_year,
                                           initialize=0)
        self.model.cost_Fund_i = pyo.Var(self.model.set_fund,
                                           initialize=0)
        self.model.cost_Tax_i_y = pyo.Var(self.model.set_tax, self.model.set_year,
                                          initialize=0)
        self.model.cost_Tax_i = pyo.Var(self.model.set_tax,
                                          initialize=0)

        self.model.cost_UEAC_i_y = pyo.Var(self.model.set_ueac, self.model.set_year,
                                           initialize=0)
        self.model.cost_UEAC_i = pyo.Var(self.model.set_ueac,
                                         initialize=0)

        self.model.u_tf_y = pyo.Var(self.model.set_tariff, self.model.set_year,
                                    within=pyo.Binary)

        """
        모델에 미 반영된 변수
        """

        # stoarge
        self.model.ch_ESS_s_y_d_h = pyo.Var(self.model.set_storage, self.model.set_year, self.model.set_day,
                                            self.model.set_hour,
                                            bounds=(0, None), initialize=0)
        self.model.dch_ESS_s_y_d_h = pyo.Var(self.model.set_storage, self.model.set_year, self.model.set_day,
                                             self.model.set_hour,
                                             bounds=(0, None), initialize=0)
        self.model.soc_ESS_s_y_d_h = pyo.Var(self.model.set_storage, self.model.set_year, self.model.set_day,
                                             self.model.set_hour,
                                             bounds=(0, None), initialize=0)

        self.model.w_s_y_d_h = pyo.Var(self.model.set_storage, self.model.set_year, self.model.set_day, self.model.set_hour,
                                       within=pyo.Binary)

        self.model.cost_Res_s = pyo.Var(self.model.set_storage, initialize=0)

        return self.model

    def define_objective_function(self):
        self.model.obj = pyo.Objective(expr=sum(self.model.cost_SG_g[g] for g in self.model.set_sg_generator) +
                                            sum(self.model.cost_PPA_i[i] for i in self.model.set_ppa_item) +
                                            sum(self.model.cost_ESS_s[s] for s in self.model.set_storage) +
                                            sum(self.model.cost_TrF_i[i] for i in self.model.set_trf_item) +
                                            sum(self.model.cost_Tax_i[i] for i in self.model.set_tax) +
                                            sum(self.model.cost_Fund_i[i] for i in self.model.set_fund) +
                                            sum(self.model.cost_UEAC_i[i] for i in self.model.set_ueac) -
                                            sum(self.model.cost_RV_SG_g[g] for g in self.model.set_sg_generator) -
                                            sum(self.model.cost_RV_ESS_s[s] for s in self.model.set_storage),
                                       sense=pyo.minimize
                                       )
        return self.model

    def define_constraints_SG(self, options: ProbOptions, ST: SetTime):
        def net_capa_SG_year_rule(model, g, y):
            """
            :return: 연도별 자가발전 순용량 [분석기간 step 용량의 총합 + 과거 누적 순용량 - 분석기간 retire 용량의 총합]
            """
            year_init = ST.set_year[0]
            return model.net_capa_SG_g_y[g, y] == \
                sum(model.step_capa_SG_g_y[g, y_tilda] for y_tilda in range(year_init, y + 1, 1)) + \
                model.para_cumul_net_capa_SG_g[g] - \
                sum(model.retire_capa_SG_g_y[g, y_tilda] for y_tilda in range(year_init, y + 1, 1))
        self.model.const_net_capa_SG_year = pyo.Constraint(self.model.set_sg_generator,
                                                           self.model.set_year,
                                                           rule=net_capa_SG_year_rule)

        def retire_capa_SG_year_rule(model, g, y):
            """
            :param y_tilda: 현재 연도 y - 자가발전기 g의 수명 [g의 수명만큼 이후의 연도에는 해당 용량만큼 퇴출됨]
            3가지 케이스로 분류
            1. y_tilda < 모형 초기연도
             :return 퇴출용량 = 0 [모형 초기연도 이전에 결정된 용량은 없음]
            2. 모형 초기연도 < y_tilda < 분석기간 초기연도
             :return 퇴출용량 = y_tilda 연도의 step 용량 [분석기간 초기연도보다 이전 연도이므로 parameter 값으로 입력]
            3. y_tilda > 분석기간 초기연도
             :return 퇴출용량 = y_tilda 연도의 step 용량 [y_tilda 연도가 결정되는 연도이므로 step 용량은 결정변수]
            """
            y_tilda = y - options.span_dict['SG'][g]
            year0 = options.year0
            year_init = ST.set_year[0]

            if y_tilda < year0:
                return model.retire_capa_SG_g_y[g, y] == 0
            elif year0 <= y_tilda < year_init:
                return model.retire_capa_SG_g_y[g, y] == model.para_past_step_capa_SG_g_y[g, y_tilda]
            else:
                return model.retire_capa_SG_g_y[g, y] == model.step_capa_SG_g_y[g, y_tilda]

        self.model.const_retire_capa_SG_year = pyo.Constraint(self.model.set_sg_generator,
                                                              self.model.set_year,
                                                              rule=retire_capa_SG_year_rule)

        def cost_inv_SG_year_rule(model, g, y):
            """
            :return: y연도 g 발전기의 투자비용 = y연도 step 용량 x y연도 CAPEX 단가
            """
            return model.cost_inv_SG_g_y[g, y] == model.step_capa_SG_g_y[g, y] * model.para_price_CPX_g_y[g, y]
        self.model.const_cost_inv_SG_year = pyo.Constraint(self.model.set_sg_generator,
                                                           self.model.set_year,
                                                           rule=cost_inv_SG_year_rule)

        def cost_oper_SG_year_rule(model, g, y):
            """
            :return: y연도 g 발전기의 운영비용 = y연도 순용량 x y연도 OPEX 단가
            """
            return model.cost_oper_SG_g_y[g, y] == model.net_capa_SG_g_y[g, y] * model.para_price_OPX_g_y[g, y]
        self.model.const_cost_oper_SG_year = pyo.Constraint(self.model.set_sg_generator,
                                                            self.model.set_year,
                                                            rule=cost_oper_SG_year_rule)

        def cost_SG_total_rule(model, g):
            """
            :return: 사회적할인율을 고려하여 g 발전기에 대한 분석기간 총 비용 산정
            """
            return model.cost_SG_g[g] == sum(model.para_Discount_y[y] *
                                             (model.cost_inv_SG_g_y[g, y] +
                                              model.cost_oper_SG_g_y[g, y])
                                             for y in model.set_year)
        self.model.const_cost_SG_total = pyo.Constraint(self.model.set_sg_generator,
                                                        rule=cost_SG_total_rule)

        def cost_SV_SG_total_rule(model, g):
            """
            :return: 사회적할인율을 고려하여 g 발전기에 대한 잔존가치 산정 [y연도 투자비용 * 분석기간 마지막연도에서 남아있는 잔존가치 비율]
            """
            return model.cost_RV_SG_g[g] == sum(model.para_Discount_y[y] *
                                                model.para_rate_RVR_SG_g_y[g, y] *
                                                model.cost_inv_SG_g_y[g, y]
                                                for y in model.set_year)
        self.model.const_cost_SV_SG_total = pyo.Constraint(self.model.set_sg_generator,
                                                           rule=cost_SV_SG_total_rule)

        def capa_SG_max_rule(model, g, y):
            """
            :return: 자가발전 순용량의 최대용량 제약
            """
            return model.net_capa_SG_g_y[g, y] <= model.para_capa_max_SG_g_y[g, y]
        self.const_capa_SG_max = pyo.Constraint(self.model.set_sg_generator,
                                                self.model.set_year,
                                                rule=capa_SG_max_rule)

        def capa_SG_min_rule(model, g, y):
            """
            :return: 자가발전 순용량의 최소용량 제약
            """
            return model.net_capa_SG_g_y[g, y] >= model.para_capa_min_SG_g_y[g, y]
        self.const_capa_SG_min = pyo.Constraint(self.model.set_sg_generator,
                                                self.model.set_year,
                                                rule=capa_SG_min_rule)

        def gen_SG_hour_rule(model, g, y, d, h):
            """
            :return: 자가발전량 = 순용량 * 시간대별 이용률
            """
            return model.p_SG_g_y_d_h[g, y, d, h] == \
                model.net_capa_SG_g_y[g, y] * model.para_factor_g_y_d_h[g, y, d, h]
        self.const_gen_SG_hour = pyo.Constraint(self.model.set_ppa_generator,
                                                self.model.set_year,
                                                self.model.set_day,
                                                self.model.set_hour,
                                                rule=gen_SG_hour_rule)

        return self.model

    def define_constraints_PPA(self, options: ProbOptions, ST: SetTime):
        def net_capa_PPA_year_rule(model, g, y):
            """
            :return: 연도별 PPA 순용량 [분석기간 step 용량의 총합 + 과거 누적 순용량 - 분석기간 retire 용량의 총합]
            """
            year_init = ST.set_year[0]
            return model.net_capa_PPA_g_y[g, y] == \
                sum(model.step_capa_PPA_g_y[g, y_tilda] for y_tilda in range(year_init, y + 1, 1)) + \
                model.para_cumul_net_capa_PPA_g[g] - \
                sum(model.retire_capa_PPA_g_y[g, y_tilda] for y_tilda in range(year_init, y + 1, 1))
        self.model.const_net_capa_PPA_year = pyo.Constraint(self.model.set_ppa_generator,
                                                            self.model.set_year,
                                                            rule=net_capa_PPA_year_rule)
        def retire_capa_PPA_year_rule(model, g, y):
            """
            :param y_tilda: 현재 연도 y - PPA 계약기간 [계약시점 y 이후로 계약기간이 지난 연도에는 계약용량만큼 퇴출됨]
            3가지 케이스로 분류
            1. y_tilda < 모형 초기연도
             :return 퇴출용량 = 0 [모형 초기연도 이전에 계약된 용량은 없음]
            2. 모형 초기연도 < y_tilda < 분석기간 초기연도
             :return 퇴출용량 = y_tilda 연도의 step 용량 [분석기간 초기연도보다 이전 연도이므로 parameter 값으로 입력]
            3. y_tilda > 분석기간 초기연도
             :return 퇴출용량 = y_tilda 연도의 step 용량 [y_tilda 연도가 결정되는 연도이므로 step 용량은 결정변수]
            """
            y_tilda = y - options.span_dict['PPA'][g]
            year0 = options.year0
            year_init = ST.set_year[0]
            if y_tilda < year0:
                return model.retire_capa_PPA_g_y[g, y] == 0
            elif year0 <= y_tilda < year_init:
                return model.retire_capa_PPA_g_y[g, y] == model.para_past_step_capa_PPA_g_y[g, y_tilda]
            else:
                return model.retire_capa_PPA_g_y[g, y] == model.step_capa_PPA_g_y[g, y_tilda]
        self.model.const_retire_capa_PPA_year = pyo.Constraint(self.model.set_ppa_generator,
                                                               self.model.set_year,
                                                               rule=retire_capa_PPA_year_rule)

        def cost_PPA_use_hour_rule(model, g, y, d, h):
            """
            1. 현재 y 연도 - 계약기간 + 1 < 모형 초기연도 [현재 y연도 기준으로 계약이 종료된 용량이 없는 경우]
             :return 시간별 PPA 비용 = y_tilda연도 step 용량 * y년도 이용률 * y_tilda연도 단가 * 축약된 일수
                                     ** y_tilda가 분석 초기연도보다 이전일 경우, para step 사용
                                     ** y_tilda가 분석 초기연도보다 이후일 경우, var step 사용
                                     ** y_tilda = 모형 초기연도 ~ y 연도
            2. 현재 y 연도 - 계약기간 + 1 >= 모형 초기연도 [현재 y연도 기준으로 계약이 종료된 용량이 있을 수도 있는 경우]
             :return 시간별 PPA 비용 = y_tilda연도 step 용량 * y년도 이용률 * y_tilda연도 단가 * 축약된 일수
                                     ** y_tilda가 분석 초기연도보다 이전일 경우, para step 사용
                                     ** y_tilda가 분석 초기연도보다 이후일 경우, var step 사용
                                     ** y_tilda = y연도 - 계약기간 + 1 ~ y연도 [y연도 기준으로 PPA 계약이 종료된 이후 시점부터 step 용량 고려함]
            """
            year0 = options.year0
            year_init = ST.set_year[0]
            if y - options.span_dict['PPA'][g] + 1 < year0:
                return model.cost_PPA_Use_g_y_d_h[g, y, d, h] == \
                    sum(model.para_past_step_capa_PPA_g_y[g, y_tilda] *
                        model.para_factor_g_y_d_h[g, y, d, h] *
                        model.para_price_PPA_Use_g_y[g, y_tilda] *
                        model.para_N_y_d_h[y, d, h]
                        if y_tilda < year_init else
                        model.step_capa_PPA_g_y[g, y_tilda] *
                        model.para_factor_g_y_d_h[g, y, d, h] *
                        model.para_price_PPA_Use_g_y[g, y_tilda] *
                        model.para_N_y_d_h[y, d, h]
                        for y_tilda in range(year0, y + 1, 1))
            else:
                return model.cost_PPA_Use_g_y_d_h[g, y, d, h] == \
                    sum(model.para_past_step_capa_PPA_g_y[g, y_tilda] *
                        model.para_factor_g_y_d_h[g, y, d, h] *
                        model.para_price_PPA_Use_g_y[g, y_tilda] *
                        model.para_N_y_d_h[y, d, h]
                        if y_tilda < year_init else
                        model.step_capa_PPA_g_y[g, y_tilda] *
                        model.para_factor_g_y_d_h[g, y, d, h] *
                        model.para_price_PPA_Use_g_y[g, y_tilda] *
                        model.para_N_y_d_h[y, d, h]
                        for y_tilda in range(y - options.span_dict['PPA'][g] + 1, y + 1, 1))
        self.model.const_cost_PPA_use_hour = pyo.Constraint(self.model.set_ppa_generator,
                                                            self.model.set_year,
                                                            self.model.set_day,
                                                            self.model.set_hour,
                                                            rule=cost_PPA_use_hour_rule)

        def cost_PPA_item_year_rule(model, g, i, y):
            """
            :return: PPA의 연도별 및 아이템별 비용 정의
                    * Use 비용을 제외한 아이템들은 y 연도 단가에 적용 받음
            """
            if i == 'Use':
                return model.cost_PPA_g_i_y[g, i, y] == sum(model.cost_PPA_Use_g_y_d_h[g, y, d, h]
                                                            for d in model.set_day
                                                            for h in model.set_hour)

            else:
                return model.cost_PPA_g_i_y[g, i, y] == sum(model.p_PPA_g_y_d_h[g, y, d, h] *
                                                            model.para_N_y_d_h[y, d, h]
                                                            for d in model.set_day
                                                            for h in model.set_hour) * \
                    model.para_price_PPA_g_i_y[g, i, y]
        self.model.const_cost_PPA_item_year = pyo.Constraint(self.model.set_ppa_generator,
                                                             self.model.set_ppa_item,
                                                             self.model.set_year,
                                                             rule=cost_PPA_item_year_rule)

        def cost_PPA_total_rule(model, i):
            """
            :return: 사회적할인율을 고려하여 아이템별 분석기간 총비용 산정
            """
            return model.cost_PPA_i[i] == sum(model.para_Discount_y[y] * model.cost_PPA_g_i_y[g, i, y]
                                              for g in model.set_ppa_generator
                                              for y in model.set_year)
        self.model.const_cost_PPA_total_rule = pyo.Constraint(self.model.set_ppa_item,
                                                              rule=cost_PPA_total_rule)

        def gen_PPA_hour_rule(model, g, y, d, h):
            """
            :return: PPA 발전량 = 순용량 * 시간대별 이용률
            """
            return model.p_PPA_g_y_d_h[g, y, d, h] == \
                model.net_capa_PPA_g_y[g, y] * model.para_factor_g_y_d_h[g, y, d, h]
        self.const_gen_PPA_hour = pyo.Constraint(self.model.set_ppa_generator,
                                                 self.model.set_year,
                                                 self.model.set_day,
                                                 self.model.set_hour,
                                                 rule=gen_PPA_hour_rule)

        def capa_PPA_max_rule(model, tf, g, y):
            """
            :return: pro 요금제가 적용될 경우(pre 요금제 사용 x), PPA 용량은 PPA 가능한 최대용량으로 제약
            """
            if tf == 'pro':
                return model.net_capa_PPA_g_y[g, y] <= model.para_capa_max_PPA_g_y[g, y] * model.u_tf_y[y]
        self.const_capa_PPA_max = pyo.Constraint(self.model.set_tariff,
                                                 self.model.set_ppa_generator,
                                                 self.model.set_year,
                                                 rule=capa_PPA_max_rule)

        def logic_PPA_bin_leq_year_rule(model, tf, y):
            """
            :return pro 요금제는 순용량이 0일 경우, 이진결정변수가 0이 되도록 disjunctive 방법론 적용
                               순용량이 0보다 클 경우, 이진결정변수가 1이 되도록 제약
            """
            if tf == 'pro':
                return model.u_tf_y[tf, y] <= \
                    sum(model.net_capa_PPA_g_y[g, y] for g in model.set_ppa_generator) * options.Big_M
        self.model.const_logic_PPA_bin_leq_year = pyo.Constraint(self.model.set_tariff,
                                                                 self.model.set_year,
                                                                 rule=logic_PPA_bin_leq_year_rule)
        def logic_PPA_bin_geq_year_rule(model, tf, y):
            if tf == 'pro':
                return model.u_tf_y[tf, y] >= \
                    sum(model.net_capa_PPA_g_y[g, y] for g in model.set_ppa_generator) / options.Big_M
        self.model.const_logic_PPA_bin_geq_year = pyo.Constraint(self.model.set_tariff,
                                                                 self.model.set_year,
                                                                 rule=logic_PPA_bin_geq_year_rule)

        return self.model

    def define_constraints_ESS(self, options: ProbOptions, ST: SetTime):
        def net_P_capa_ESS_year_rule(model, s, y):
            """
            :return: 연도별 ESS 순용량 [분석기간 step 용량의 총합 + 과거 누적 순용량 - 분석기간 retire 용량의 총합]
            """
            year_init = ST.set_year[0]
            return model.net_P_capa_ESS_s_y[s, y] == \
                sum(model.step_P_capa_ESS_s_y[s, y_tilda] for y_tilda in range(year_init, y + 1, 1)) + \
                model.para_cumul_net_P_capa_ESS_s[s] - \
                sum(model.retire_P_capa_ESS_s_y[s, y_tilda] for y_tilda in range(year_init, y + 1, 1))

        self.model.const_net_capa_ESS_year = pyo.Constraint(self.model.set_storage,
                                                            self.model.set_year,
                                                            rule=net_P_capa_ESS_year_rule)
        def retire_P_capa_ESS_year_rule(model, s, y):
            """
            :param y_tilda: 현재 연도 y - ESS 수명
            3가지 케이스로 분류
            1. y_tilda < 모형 초기연도
             :return 퇴출용량 = 0 [모형 초기연도 이전에 설치된 용량은 없음]
            2. 모형 초기연도 < y_tilda < 분석기간 초기연도
             :return 퇴출용량 = y_tilda 연도의 step 용량 [분석기간 초기연도보다 이전 연도이므로 parameter 값으로 입력]
            3. y_tilda > 분석기간 초기연도
             :return 퇴출용량 = y_tilda 연도의 step 용량 [y_tilda 연도가 결정되는 연도이므로 step 용량은 결정변수]
            """
            y_tilda = y - options.span_dict['Bat'][s]
            year0 = options.year0
            year_init = ST.set_year[0]

            if y_tilda < year0:
                return model.retire_P_capa_ESS_s_y[s, y] == 0
            elif year0 <= y_tilda < year_init:
                return model.retire_P_capa_ESS_s_y[s, y] == model.para_past_step_P_capa_ESS_s_y[s, y_tilda]
            elif y_tilda >= year_init:
                return model.retire_P_capa_ESS_s_y[s, y] == model.step_P_capa_ESS_s_y[s, y_tilda]
            else:
                pass
        self.model.const_retire_P_capa_ESS_year = pyo.Constraint(self.model.set_storage,
                                                                 self.model.set_year,
                                                                 rule=retire_P_capa_ESS_year_rule)

        def net_E_capa_ESS_year_rule(model, s, y):
            """
            :return: ESS의 순 에너지용량 = ESS의 순 전력용량 X 미리 정해진 C_rate
            """
            return model.net_E_capa_ESS_s_y[s, y] == model.net_P_capa_ESS_s_y[s, y] * options.storages_dict[s]
        self.model.const_net_E_capa_ESS_year = pyo.Constraint(self.model.set_storage,
                                                              self.model.set_year,
                                                              rule=net_E_capa_ESS_year_rule)

        def step_E_capa_ESS_year_rule(model, s, y):
            """
            :return: ESS의 step 에너지용량 = ESS의 step 전력용량 X 미리 정해진 C_rate
            """
            return model.step_E_capa_ESS_s_y[s, y] == model.step_P_capa_ESS_s_y[s, y] * options.storages_dict[s]
        self.model.const_step_E_capa_ESS_year = pyo.Constraint(self.model.set_storage,
                                                               self.model.set_year,
                                                               rule=step_E_capa_ESS_year_rule)

        def cost_inv_ESS_year_rule(model, s, y):
            """
            :return: 연도별 저장장치 s의 투지비용 = step 전력용량 * 전력용량 단가 + step 에너지용량 * 에너지용량 단가
            """
            return model.cost_inv_ESS_s_y[s, y] == \
                model.step_P_capa_ESS_s_y[s, y] * model.para_price_inv_ESS_P_capa_s_y[s, y] + \
                model.step_E_capa_ESS_s_y[s, y] * model.para_price_inv_ESS_E_capa_s_y[s, y]
        self.model.const_cost_inv_Bat_year = pyo.Constraint(self.model.set_storage,
                                                            self.model.set_year,
                                                            rule=cost_inv_ESS_year_rule)

        def cost_oper_ESS_year_rule(model, s, y):
            """
            :return: 연도별 저장장치 s의 운영비용 = 순전력용량 * 전력용량 운영단가 + 순에너지용량 * 에너지용량 운영단가
            """
            return model.cost_oper_ESS_s_y[s, y] == \
                model.net_P_capa_ESS_s_y[s, y] * model.para_price_oper_ESS_P_capa_s_y[s, y] + \
                model.net_E_capa_ESS_s_y[s, y] * model.para_price_oper_ESS_E_capa_s_y[s, y]
        self.model.const_cost_oper_ESS_year = pyo.Constraint(self.model.set_storage,
                                                             self.model.set_year,
                                                             rule=cost_oper_ESS_year_rule)

        def cost_ESS_total_rule(model, s):
            """
            :return: 저장장치별 분석기간 총비용 = 사회적할인율을 적용하여 투자비용과 운영비용의 합
            """
            return model.cost_ESS_s[s] == sum(model.para_Discount_y[y] *
                                              (model.cost_inv_ESS_s_y[s, y] + model.cost_oper_ESS_s_y[s, y])
                                              for y in model.set_year)
        self.model.const_cost_ESS_total = pyo.Constraint(self.model.set_storage,
                                                         rule=cost_ESS_total_rule)

        def cost_SV_ESS_total_rule(model, s):
            """
            :return: 사회적할인율을 고려하여 s 저장장치에 대한 잔존가치 산정 [y연도 투자비용 * 분석기간 마지막연도에서 남아있는 잔존가치 비율]
            """
            return model.cost_RV_ESS_s[s] == sum(model.para_Discount_y[y] *
                                                 model.para_rate_RVR_ESS_s_y[s, y] *
                                                 model.cost_inv_ESS_s_y[s, y]
                                                 for y in model.set_year)
        self.model.const_cost_SV_ESS_total = pyo.Constraint(self.model.set_sg_generator,
                                                            rule=cost_SV_ESS_total_rule)

        def ch_ESS_limit_hour_rule(model, s, y, d, h):
            """
            :return: 시간별 충전량 <= 순 전력용량
            """
            return model.ch_ESS_s_y_d_h[s, y, d, h] <= model.net_P_capa_ESS_s_y[s, y]
        self.model.const_ch_ESS_limit_hour = pyo.Constraint(self.model.set_storage,
                                                            self.model.set_year,
                                                            self.model.set_day,
                                                            self.model.set_hour,
                                                            rule=ch_ESS_limit_hour_rule)
        def dch_ESS_limit_hour_rule(model, s, y, d, h):
            """
            :return: 시간별 방전량 <= 순 전력용량
            """
            return model.dch_ESS_s_y_d_h[s, y, d, h] <= model.net_P_capa_ESS_s_y[s, y]
        self.model.const_dch_ESS_limit_hour = pyo.Constraint(self.model.set_storage,
                                                             self.model.set_year,
                                                             self.model.set_day,
                                                             self.model.set_hour,
                                                             rule=dch_ESS_limit_hour_rule)

        def soc_ESS_limit_upper_hour_rule(model, s, y, d, h):
            """
            :return: 시간별 충전상태 <= 순 에너지용량 * SOC 최대 비율
            """
            return model.soc_ESS_s_y_d_h[s, y, d, h] <= \
                model.net_E_capa_ESS_s_y[s, y] * model.para_rate_SOC_upper_ESS_s[s]
        self.model.const_soc_ESS_limit_upper_hour = pyo.Constraint(self.model.set_storage,
                                                                   self.model.set_year,
                                                                   self.model.set_day,
                                                                   self.model.set_hour,
                                                                   rule=soc_ESS_limit_upper_hour_rule)

        def soc_ESS_limit_lower_hour_rule(model, s, y, d, h):
            """
            :return: 시간별 충전상태 >= 순 에너지용량 * SOC 최소 비율
            """
            return model.soc_ESS_s_y_d_h[s, y, d, h] >= \
                model.net_E_capa_ESS_s_y[s, y] * model.para_rate_SOC_lower_ESS_s[s]
        self.model.const_soc_ESS_limit_lower_hour = pyo.Constraint(self.model.set_storage,
                                                                   self.model.set_year,
                                                                   self.model.set_day,
                                                                   self.model.set_hour,
                                                                   rule=soc_ESS_limit_lower_hour_rule)

        def soc_general_hour_rule(model, s, y, d, h):
            """
            1. d = 1 & h = 1
                :return soc 상태 = 자가방전율 * 슌에너지용량 * 초기 SOC 비율 + 충전효율이 적용되어 더 적은 충전량 - 방전효율이 적용되어 더 많은 방전량
            2. h = 1
                :return soc 상태 = 전일 마지막 시간의 soc에 자가방전율 + 충전효율이 적용되어 더 적은 충전량 - 방전효율이 적용되어 더 많은 방전량
            3. 그 외
                :return soc 상태 = 전 시간의 soc에 자가방전율 + 충전효율이 적용되어 더 적은 충전량 - 방전효율이 적용되어 더 많은 방전량
            """
            h_last = ST.set_hour[-1]
            if d == 1 and h == 1:
                return model.soc_ESS_s_y_d_h[s, y, d, h] == \
                    model.para_rate_SelfDchg_ESS_s[s] * model.net_E_capa_ESS_s_y[s, y] * model.para_rate_SOC0_ESS_s[s] + \
                    model.para_rate_ChrgEff_ESS_s[s] * model.ch_ESS_s_y_d_h[s, y, d, h] - \
                    model.para_rate_DchgEff_inv_ESS_s[s] * model.dch_ESS_s_y_d_h[s, y, d, h]

            elif h == 1:
                return model.soc_ESS_s_y_d_h[s, y, d, h] == \
                    model.para_rate_SelfDchg_ESS_s[s] * model.soc_ESS_s_y_d_h[s, y, d - 1, h_last] + \
                    model.para_rate_ChrgEff_ESS_s[s] * model.ch_ESS_s_y_d_h[s, y, d, h] - \
                    model.para_rate_DchgEff_inv_ESS_s[s] * model.dch_ESS_s_y_d_h[s, y, d, h]

            else:
                return model.soc_ESS_s_y_d_h[s, y, d, h] == \
                    model.para_rate_SelfDchg_ESS_s[s] * model.soc_ESS_s_y_d_h[s, y, d, h - 1] + \
                    model.para_rate_ChrgEff_ESS_s[s] * model.ch_ESS_s_y_d_h[s, y, d, h] - \
                    model.para_rate_DchgEff_inv_ESS_s[s] * model.dch_ESS_s_y_d_h[s, y, d, h]
        self.model.const_soc_general_hour = pyo.Constraint(self.model.set_storage,
                                                           self.model.set_year,
                                                           self.model.set_day,
                                                           self.model.set_hour,
                                                           rule=soc_general_hour_rule)

        def ch_ESS_state_hour_rule(model, s, y, d, h):
            """
            :return: 시간별 충전량 <= 충방전상태변수 * Big-M
            """
            return model.ch_ESS_s_y_d_h[s, y, d, h] <= model.w_s_y_d_h[s, y, d, h] * model.para_Big_M
        self.model.const_ch_ESS_state_hour = pyo.Constraint(self.model.set_storage,
                                                            self.model.set_year,
                                                            self.model.set_day,
                                                            self.model.set_hour,
                                                            rule=ch_ESS_state_hour_rule)
        def dch_ESS_state_hour_rule(model, s, y, d, h):
            """
            :return: 시간별 방전량 <= (1-충방전상태변수) * Big-M
            """
            return model.dch_ESS_s_y_d_h[s, y, d, h] <= (1 - model.w_s_y_d_h[s, y, d, h]) * model.para_Big_M
        self.model.const_dch_ESS_state_hour = pyo.Constraint(self.model.set_storage,
                                                             self.model.set_year,
                                                             self.model.set_day,
                                                             self.model.set_hour,
                                                             rule=dch_ESS_state_hour_rule)

        return self.model

    def define_constraints_TrF(self):
        def cost_TrF_capa_leq_year_rule(model, tf, y):
            """
            보완공급 용량은 순최대전력(결정변수)이므로 이진결정변수를 곱할 수가 없음
            disjunctive method를 적용해서 요금제에 따라서 비용이 결정되도록 유도
            :return 연간 보완공급 용량요금 = 보완공급 순최대전력 * 단가 * 12
            """
            return model.cost_TrF_Capa_y[y] * (1 / model.para_price_TrF_Capa_tf_y[tf, y]) - model.capa_CS_y[y] * 12 <= \
                (1 - model.u_tf_y[tf, y]) * model.para_Big_M
        self.model.const_cost_TrF_capa_leq_year = pyo.Constraint(self.model.set_tariff,
                                                                 self.model.set_year,
                                                                 rule=cost_TrF_capa_leq_year_rule)

        def cost_TrF_capa_geq_year_rule(model, tf, y):
            return model.cost_TrF_Capa_y[y] * (1 / model.para_price_TrF_Capa_tf_y[tf, y]) - model.capa_CS_y[y] * 12 >= \
                - (1 - model.u_tf_y[tf, y]) * model.para_Big_M
        self.model.const_cost_TrF_capa_geq_year = pyo.Constraint(self.model.set_tariff,
                                                                 self.model.set_year,
                                                                 rule=cost_TrF_capa_geq_year_rule)

        def cost_TrF_use_leq_hour_rule(model, tf, y, d, h):
            """
            disjunctive method를 적용해서 요금제에 따라서 비용이 결정되도록 유도
            :return 시간별 보완공급 사용요금 = 보완공급 사용량 * 단가 * 축약된 일수
            """
            return model.cost_TrF_Use_y_d_h[y, d, h] * (1 / model.para_price_TrF_Use_tf_y_d_h[tf, y, d, h]) - \
                model.p_TrF_y_d_h[y, d, h] * model.para_N_y_d_h[y, d, h] <= \
                (1 - model.u_tf_y[tf, y]) * model.para_Big_M
        self.model.const_cost_TrF_use_leq_hour = pyo.Constraint(self.model.set_tariff,
                                                                self.model.set_year,
                                                                self.model.set_day,
                                                                self.model.set_hour,
                                                                rule=cost_TrF_use_leq_hour_rule)

        def cost_TrF_use_geq_hour_rule(model, tf, y, d, h):
            return model.cost_TrF_Use_y_d_h[y, d, h] * (1 / model.para_price_TrF_Use_tf_y_d_h[tf, y, d, h]) - \
                model.p_TrF_y_d_h[y, d, h] * model.para_N_y_d_h[y, d, h] >= \
                - (1 - model.u_tf_y[tf, y]) * model.para_Big_M
        self.model.const_cost_TrF_use_geq_hour = pyo.Constraint(self.model.set_tariff,
                                                                self.model.set_year,
                                                                self.model.set_day,
                                                                self.model.set_hour,
                                                                rule=cost_TrF_use_geq_hour_rule)

        def cost_TrF_item_year_rule(model, i, y):
            """
            :return: 연도별 및 아이템별 보완공급 비용 (축약된 일수 고려)
            """
            if i == 'Use':
                return model.cost_TrF_i_y[i, y] == sum(model.cost_TrF_Use_y_d_h[y, d, h]
                                                       for d in model.set_day
                                                       for h in model.set_hour)
            elif i == 'Capa':
                return model.cost_TrF_i_y[i, y] == model.cost_TrF_Capa_y[y]

            else:
                return model.cost_TrF_i_y[i, y] == sum(model.p_TrF_y_d_h[y, d, h] *
                                                       model.para_N_y_d_h[y, d, h]
                                                       for d in model.set_day
                                                       for h in model.set_hour) * \
                model.para_price_TrF_i_y[i, y]
        self.model.const_cost_TrF_item_year = pyo.Constraint(self.model.set_trf_item,
                                                             self.model.set_year,
                                                             rule=cost_TrF_item_year_rule)

        def cost_TrF_total_rule(model, i):
            """
            :return: 아이템별 분석기간 전체 비용 산정
            """
            return model.cost_TrF_i[i] == sum(model.para_Discount_y[y] * model.cost_TrF_i_y[i, y]
                                              for y in model.set_year)
        self.model.const_cost_TrF_total_rule = pyo.Constraint(self.model.set_trf_item,
                                                              rule=cost_TrF_total_rule)

        def capa_TrF_year_rule(model, y, d, h):
            """
            :return: 보완공급 순최대용량 >= 수요 - 자가발전량
            """
            return model.capa_CS_y[y] >= \
                model.para_DmD_y_d_h[y, d, h] - \
                sum(model.p_SG_g_y_d_h[g, y, d, h] for g in model.set_sg_generator)
        self.const_capa_TrF_year = pyo.Constraint(self.model.set_year,
                                                  self.model.set_day,
                                                  self.model.set_hour,
                                                  rule=capa_TrF_year_rule)

        return self.model

    def define_constraints_others(self, options: ProbOptions):
        def cost_Tax_year_rule(model, i, y):
            if i == 'PPA':
                return model.cost_Tax_i_y[i, y] == sum(model.cost_PPA_g_i_y[g, j, y]
                                                       for g in model.set_ppa_generator
                                                       for j in model.set_ppa_item) * \
                    model.para_rate_Tax_i_y[i, y]
            elif i == 'TrF':
                return model.cost_Tax_i_y[i, y] == sum(model.cost_TrF_i_y[j, y]
                                                       for j in model.set_trf_item) * \
                    model.para_rate_Tax_i_y[i, y]
        self.model.const_cost_Tax_year = pyo.Constraint(self.model.set_tax,
                                                        self.model.set_year,
                                                        rule=cost_Tax_year_rule)

        def cost_Tax_total_rule(model, i):
            return model.cost_Tax_i[i] == sum(model.para_Discount_y[y] * model.cost_Tax_i_y[i, y]
                                              for y in model.set_year)
        self.model.const_Tax_total_rule = pyo.Constraint(self.model.set_tax,
                                                         rule=cost_Tax_total_rule)

        def cost_Fund_year_rule(model, i, y):
            if i == 'PPA':
                return model.cost_Fund_i_y[i, y] == sum(model.cost_PPA_g_i_y[g, j, y]
                                                        for g in model.set_ppa_generator
                                                        for j in model.set_ppa_item) * \
                    model.para_rate_Fund_i_y[i, y]
            elif i == 'TrF':
                return model.cost_Fund_i_y[i, y] == sum(model.cost_TrF_i_y[j, y]
                                                        for j in model.set_trf_item) * \
                    model.para_rate_Fund_i_y[i, y]
        self.model.const_cost_Fund_year = pyo.Constraint(self.model.set_fund,
                                                         self.model.set_year,
                                                         rule=cost_Fund_year_rule)

        def cost_Fund_total_rule(model, i):
            return model.cost_Fund_i[i] == sum(model.para_Discount_y[y] * model.cost_Fund_i_y[i, y]
                                               for y in model.set_year)
        self.model.const_Fund_total = pyo.Constraint(self.model.set_fund,
                                                     rule=cost_Fund_total_rule)

        def cost_UEAC_year_rule(model, i, y):
            return model.cost_UEAC_i_y[i, y] == model.p_UEAC_i_y[i, y] * model.para_price_UEAC_i_y[i, y]
        self.model.cost_UEAC_year = pyo.Constraint(self.model.set_ueac,
                                                   self.model.set_year,
                                                   rule=cost_UEAC_year_rule)

        def cost_UEAC_total_rule(model, i):
            return model.cost_UEAC_i[i] == sum(model.para_Discount_y[y] * model.cost_UEAC_i_y[i, y]
                                               for y in model.set_year)
        self.model.cost_UEAC_total = pyo.Constraint(self.model.set_ueac,
                                                    rule=cost_UEAC_total_rule)

        def rate_RE100_goal_year_rule(model, y):
            """
            보완공급 조달량은 재생E 조달이 아님. 따라서, UEAC로 반드시 커버해야 함
            보완공급 조달량에서 UEAC를 제외한 부분을 활용해서 RE100 달성율을 확인할 수 있음
            다음 수요공급 제약에서 수요량을 충족하기 때문에 보완공급이 줄어든다면 다른 수단으로 채워질 것으로 유도할 수 있음
            :return: 1 - (보완공급 조달량 - UEAC 구매량) / 수요량 == RE100 달성율
            """
            if options.opt_RE100_achievement == 'Fix':
                return 1 - (
                        (sum(model.p_TrF_y_d_h[y, d, h]
                             for d in model.set_day
                             for h in model.set_hour) -
                         sum(model.p_UEAC_i_y[i, y]
                             for i in model.set_ueac)
                         ) /
                        sum(model.para_DmD_y_d_h[y, d, h]
                            for d in model.set_day
                            for h in model.set_hour)) == model.para_rate_RE100_y[y]
            if options.opt_RE100_achievement == 'Free':
                return 1 - (
                        (sum(model.p_TrF_y_d_h[y, d, h]
                             for d in model.set_day
                             for h in model.set_hour) -
                         sum(model.p_UEAC_i_y[i, y]
                             for i in model.set_ueac)
                         ) /
                        sum(model.para_DmD_y_d_h[y, d, h]
                            for d in model.set_day
                            for h in model.set_hour)) >= model.para_rate_RE100_y[y]
        self.const_rate_RE100_goal_year = pyo.Constraint(self.model.set_year,
                                                         rule=rate_RE100_goal_year_rule)

        def gen_DmD_min_hour_rule(model, y, d, h):
            """
            :return: 자가발전량 + PPA 조달량 + ESS 방전량 - ESS 충전량 + 보완공급 조달량 >= 수요량
            """
            return \
                    sum(model.p_SG_g_y_d_h[g, y, d, h] for g in model.set_sg_generator) + \
                    sum(model.p_PPA_g_y_d_h[g, y, d, h] for g in model.set_ppa_generator) + \
                    sum(model.dch_ESS_s_y_d_h[s, y, d, h] for s in model.set_storage) - \
                    sum(model.ch_ESS_s_y_d_h[s, y, d, h] for s in model.set_storage) + \
                    model.p_TrF_y_d_h[y, d, h] >= \
                    model.para_DmD_y_d_h[y, d, h]
        self.const_gen_DmD_min_hour = pyo.Constraint(self.model.set_year,
                                                     self.model.set_day,
                                                     self.model.set_hour,
                                                     rule=gen_DmD_min_hour_rule)

        def logic_bin_year_rule(model, y):
            """
            :return 요금제 선택 기준 이진결정변수의 총합 = 1
            """
            return sum(model.u_tf_y[tf, y] for tf in model.set_tariff) == 1
        self.model.const_logic_bin_year = pyo.Constraint(self.model.set_year,
                                                         rule=logic_bin_year_rule)

        return self.model

    def print_message(self):
        print("Construction of model is completed")


def setting_parameters(input_model: PortfolioModel,
                       options: ProbOptions,
                       input_parameters_pyomo: ParameterPyomoForm,
                       ST: SetTime,
                       net_capa_SG_g_y, net_capa_PPA_g_y, step_capa_SG_g_y, step_capa_PPA_g_y
                       ):

    for y in ST.set_historic_year:
        for g in input_parameters_pyomo.set_ppa_generator:
            input_model.para_price_PPA_Use_g_y[g, y] = input_parameters_pyomo.price_PPA_Use_g_y[g][y]

    for y in ST.set_year:
        input_model.para_Discount_y[y] = input_parameters_pyomo.rate_discount_y[y]
        input_model.para_rate_RE100_y[y] = input_parameters_pyomo.rate_achievement_y[y]

        for d in ST.set_day:
            for h in ST.set_hour:
                input_model.para_DmD_y_d_h[y, d, h] = input_parameters_pyomo.demand_y_d_h[y, d, h]

        """
        SG
        """
        for g in input_parameters_pyomo.set_sg_generator:
            input_model.para_capa_max_SG_g_y[g, y] = input_parameters_pyomo.capa_max_SG_g_y[g][y]
            input_model.para_capa_min_SG_g_y[g, y] = input_parameters_pyomo.capa_min_SG_g_y[g][y]
            input_model.para_price_CPX_g_y[g, y] = input_parameters_pyomo.price_CPX_g_y[g][y]
            input_model.para_price_OPX_g_y[g, y] = input_parameters_pyomo.price_OPX_g_y[g][y]

        for g in input_parameters_pyomo.set_ppa_generator:
            input_model.para_capa_max_PPA_g_y[g, y] = input_parameters_pyomo.capa_max_PPA_g_y[g][y]
            input_model.para_price_PPA_Use_g_y[g, y] = input_parameters_pyomo.price_PPA_Use_g_y[g][y]
            input_model.para_price_PPA_g_i_y[g, 'NT_use', y] = input_parameters_pyomo.price_NT_Use_y[y]
            input_model.para_price_PPA_g_i_y[g, 'NT_capa', y] = input_parameters_pyomo.price_NT_Capa_y[y]
            input_model.para_price_PPA_g_i_y[g, 'Uplift', y] = input_parameters_pyomo.price_Uplift_y[y]
            input_model.para_price_PPA_g_i_y[g, 'TLT', y] = input_parameters_pyomo.price_TLT_y[y]

        """
        TrF
        """
        input_model.para_price_TrF_Capa_tf_y['pre', y] = input_parameters_pyomo.price_TrF_Capa_pre_y[y]
        input_model.para_price_TrF_Capa_tf_y['pro', y] = input_parameters_pyomo.price_TrF_Capa_pro_y[y]
        for d in ST.set_day:
            for h in ST.set_hour:
                input_model.para_price_TrF_Use_tf_y_d_h['pre', y, d, h] = input_parameters_pyomo.price_TrF_Use_pre_y_d_h[y, d, h]
                input_model.para_price_TrF_Use_tf_y_d_h['pro', y, d, h] = input_parameters_pyomo.price_TrF_Use_pro_y_d_h[y, d, h]
        input_model.para_price_TrF_i_y['CCEC', y] = input_parameters_pyomo.price_CCEC_y[y]
        input_model.para_price_TrF_i_y['FCPTAR', y] = input_parameters_pyomo.price_FCPTAR_y[y]

        """
        UEAC
        """
        input_model.para_price_UEAC_i_y['REC', y] = input_parameters_pyomo.price_REC_y[y]

        """
        ESS
        """



        """
        Tax & Fund
        """
        input_model.para_rate_Tax_i_y['PPA', y] = input_parameters_pyomo.rate_Tax_y[y]
        input_model.para_rate_Tax_i_y['TrF', y] = input_parameters_pyomo.rate_Tax_y[y]
        input_model.para_rate_Fund_i_y['PPA', y] = input_parameters_pyomo.rate_Fund_y[y]
        input_model.para_rate_Fund_i_y['TrF', y] = input_parameters_pyomo.rate_Fund_y[y]

        """
        Factor, RVR
        """
        for g in input_parameters_pyomo.set_generator:
            for d in ST.set_day:
                for h in ST.set_hour:
                    input_model.para_factor_g_y_d_h[g, y, d, h] = input_parameters_pyomo.factor_g_y_d_h[g][y, d, h]

        for g in input_parameters_pyomo.set_sg_generator:
            input_model.para_rate_RVR_SG_g_y[g, y] = input_parameters_pyomo.rate_RVR_SG_g_y[g][y]

        for d in ST.set_day:
            for h in ST.set_hour:
                input_model.para_N_y_d_h[y, d, h] = input_parameters_pyomo.N_y_d_h[y, d, h]

        input_model.para_Big_M = options.Big_M

    for g in input_parameters_pyomo.set_sg_generator:
        input_model.para_cumul_net_capa_SG_g[g] = \
            sum(net_capa_SG_g_y[g, y] for y in range(options.year0, ST.set_year[0], 1))
        for y in ST.set_historic_year:
            input_model.para_past_step_capa_SG_g_y[g, y] = step_capa_SG_g_y[g, y]

    for g in input_parameters_pyomo.set_ppa_generator:
        input_model.para_cumul_net_capa_PPA_g[g] = \
            sum(net_capa_PPA_g_y[g, y] for y in range(options.year0, ST.set_year[0], 1))
        for y in ST.set_historic_year:
            input_model.para_past_step_capa_PPA_g_y[g, y] = step_capa_PPA_g_y[g, y]

    print("parameter setting is completed")
    return input_model

def solve_optimal_portfolio(options: ProbOptions, input_parameters_pyomo: ParameterPyomoForm, ST: SetTime,
                            solver_name,
                            net_capa_SG_g_y, net_capa_PPA_g_y, step_capa_SG_g_y, step_capa_PPA_g_y):

    portfolio_model = PortfolioModel(options, input_parameters_pyomo, ST)
    portfolio_model.model = setting_parameters(portfolio_model.model, options, input_parameters_pyomo, ST,
                                               net_capa_SG_g_y, net_capa_PPA_g_y, step_capa_SG_g_y, step_capa_PPA_g_y)

    instance = portfolio_model.model.create_instance()

    solver_name = solver_name
    print("solving the problem using", solver_name)
    solver = pyo.SolverFactory(solver_name)
    solver.options['workmem'] = 20480
    solver.options['mip tolerances mipgap'] = options.gaprel

    if options.solver_message:
        solver.solve(instance, tee=True)
    else:
        solver.solve(instance)

    return instance
