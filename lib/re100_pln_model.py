import pulp as lp
import numpy as np
import copy

from lib.parameters import ParameterPulpFrom, ReadInputData, ProbOptions


# Next group case
def solve_re100_milp(options: ProbOptions, input_parameters_pulp: ParameterPulpFrom, solver_name):
    if options.customize_exemption_flag:
        zero_value = input_parameters_pulp.fill_value_to_dict(0)
        lambda_nt_d_y, lambda_nt_c_y = zero_value, zero_value
        lambda_AS_payment_y = zero_value
        lambda_loss_payment_y = zero_value
        ratio_ppa_funding_y = zero_value
        ratio_commission_ppa_ratio_per_won_y = zero_value
        
    else:
        lambda_nt_d_y = input_parameters_pulp.lambda_nt_d_y
        lambda_nt_c_y = input_parameters_pulp.lambda_nt_c_y
        lambda_AS_payment_y = input_parameters_pulp.lambda_AS_payment_y
        lambda_loss_payment_y = input_parameters_pulp.lambda_loss_payment_y
        ratio_ppa_funding_y = input_parameters_pulp.ratio_ppa_funding_y
        ratio_commission_ppa_ratio_per_won_y = input_parameters_pulp.ratio_commission_ppa_ratio_per_won_y

    """
    인풋데이터 입력
    """
    set_y = input_parameters_pulp.set_y
    set_d = input_parameters_pulp.set_d
    set_h = input_parameters_pulp.set_h

    set_y_d_h = [(y, d, h) for y in set_y for d in set_d for h in set_h]
    set_d_h = [(d, h) for d in set_d for h in set_h]

    # 재생e 이용률
    Apv_d_h = input_parameters_pulp.Apv_d_h
    Aonshore_d_h = input_parameters_pulp.Aonshore_d_h

    # 부하 pattern
    demand_y_d_h = input_parameters_pulp.demand_y_d_h
    # 에너지효율화 pattern
    ee_y_d_h = input_parameters_pulp.ee_y_d_h
    lambda_ee_y = input_parameters_pulp.lambda_ee_y

    # 자가발전 관련 가격
    lambda_CAPEX_PV_y = input_parameters_pulp.lambda_CAPEX_PV_y
    lambda_OPEX_PV_y = input_parameters_pulp.lambda_OPEX_PV_y
    lambda_CAPEX_onshore_y = input_parameters_pulp.lambda_CAPEX_onshore_y
    lambda_OPEX_onshore_y = input_parameters_pulp.lambda_OPEX_onshore_y
    lambda_tariff_fixed_won_per_kW = input_parameters_pulp.lambda_tariff_fixed_won_per_kW

    # 설치 가능한 최대, 최소 용량
    cap_max_sg_pv_y = input_parameters_pulp.cap_max_sg_pv_y
    cap_max_sg_onshore_y = input_parameters_pulp.cap_max_sg_onshore_y
    cap_min_sg_pv_y = input_parameters_pulp.cap_min_sg_pv_y
    cap_min_sg_onshore_y = input_parameters_pulp.cap_min_sg_onshore_y

    # 계약 가능한 최대용량
    cap_max_ppa_pv_y = input_parameters_pulp.cap_max_ppa_pv_y
    cap_max_ppa_onshore_y = input_parameters_pulp.cap_max_ppa_onshore_y

    # 보완공급 관련 가격
    lambda_tariff_y_d_h = input_parameters_pulp.lambda_tariff_y_d_h
    lambda_climate_y = input_parameters_pulp.lambda_climate_y
    lambda_fuel_adjustment_y = input_parameters_pulp.lambda_fuel_adjustment_y

    # 기업PPA 계약 관련 가격
    lambda_PPA_pv_y = input_parameters_pulp.lambda_PPA_pv_y
    lambda_PPA_onshore_y = input_parameters_pulp.lambda_PPA_onshore_y

    lambda_welfare_y = input_parameters_pulp.lambda_welfare_y

    # 전력거래 관련 부가요소
    ratio_tariff_funding_y = input_parameters_pulp.ratio_tariff_funding_y

    ratio_commission_tariff_ratio_per_won_y = input_parameters_pulp.ratio_commission_tariff_ratio_per_won_y

    # 인증서 관련 가격
    lambda_eac_y = input_parameters_pulp.lambda_eac_y

    # simulation 관련 요소
    achievement_ratio_y = options.achievement_ratio_y
    discount_rate = input_parameters_pulp.discount_rate
    pv_life_span = input_parameters_pulp.pv_life_span
    onshore_life_span = input_parameters_pulp.onshore_life_span
    gaprel = options.gaprel
    load_cap = options.load_cap

    """
    결정변수 선언
    """
    p_tariff_y_d_h = lp.LpVariable.dicts("p_tariff_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')
    p_ppa_pv_y_d_h = lp.LpVariable.dicts("p_ppa_pv_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')
    p_ppa_onshore_y_d_h = lp.LpVariable.dicts("p_ppa_onshore_y_d_h", set_y_d_h, lowBound=0, cat='Continuous')

    p_eac_y = lp.LpVariable.dicts("p_eac_y", set_y, lowBound=0, cat='Continuous')
    capacity_pv_y = lp.LpVariable.dicts("capacity_pv_y", set_y, lowBound=0, cat='Continuous')
    capacity_onshore_y = lp.LpVariable.dicts("capacity_onshore_y", set_y, lowBound=0, cat='Continuous')

    c_sg_y = lp.LpVariable.dicts("c_sg_y", set_y, lowBound=0, cat='Continuous')
    c_tariff_y = lp.LpVariable.dicts("c_tariff_y", set_y, lowBound=0, cat='Continuous')
    c_ppa_y = lp.LpVariable.dicts("c_ppa_y", set_y, lowBound=0, cat='Continuous')
    c_eac_y = lp.LpVariable.dicts("c_eac_y", set_y, lowBound=0, cat='Continuous')
    c_residual_y = lp.LpVariable.dicts("c_residual_y", set_y, lowBound=0, cat='Continuous')
    c_loss_payment_y = lp.LpVariable.dicts("c_loss_payment_y", set_y, lowBound=0, cat='Continuous')
    c_funding_tariff_y = lp.LpVariable.dicts("c_funding_tariff_y", set_y, lowBound=0, cat='Continuous')
    c_funding_ppa_y = lp.LpVariable.dicts("c_funding_ppa_y", set_y, lowBound=0, cat='Continuous')
    c_commission_kepco_y = lp.LpVariable.dicts("c_commission_kepco_y", set_y, lowBound=0, cat='Continuous')
    c_commission_ppa_y = lp.LpVariable.dicts("c_commission_ppa_y", set_y, lowBound=0, cat='Continuous')
    # c_commission_y = lp.LpVariable.dicts("c_commission_y", set_y, lowBound=0, cat='Continuous')
    c_ppa_network_basic_y = lp.LpVariable.dicts("c_ppa_network_basic_y", set_y, lowBound=0, cat='Continuous')

    u_y = lp.LpVariable.dicts("u_y", set_y, lowBound=0, cat='Binary')   # PPA 선택유무

    model = lp.LpProblem("RE100_Problem", lp.LpMinimize)

    # objective function
    model += lp.lpSum((c_sg_y[y] + c_tariff_y[y] + c_ppa_y[y] + c_ppa_network_basic_y[y] + c_eac_y[y] +
                       c_loss_payment_y[y] + c_funding_tariff_y[y] + c_funding_ppa_y[y] + c_commission_kepco_y[y] + c_commission_ppa_y[y] - c_residual_y[y] + lambda_ee_y[y]) /
                      (1 + discount_rate) ** (y - options.present_year) for y in set_y)

    # constraints
    for y in set_y:
        y_tilda = options.year1 - y + 1
        if y == options.year0:
            # 자가발전 비용
            model += c_sg_y[y] == capacity_pv_y[y] * (lambda_CAPEX_PV_y[y] + lambda_OPEX_PV_y[y]) + \
                     capacity_onshore_y[y] * (lambda_CAPEX_onshore_y[y] + lambda_OPEX_onshore_y[y])

            if sum(demand_y_d_h[y, d, h] for d, h in set_d_h) > 0:
                # 기업PPA 비용
                model += c_ppa_y[y] == lp.lpSum(p_ppa_pv_y_d_h[y, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_pv_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_onshore_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y_y[y])
            else:
                pass

            # 잔존가치 비용
            model += c_residual_y[y] == lambda_CAPEX_PV_y[y] * capacity_pv_y[y] * (pv_life_span - y_tilda) / pv_life_span + \
                     lambda_CAPEX_onshore_y[y] * capacity_onshore_y[y] * (onshore_life_span - y_tilda) / onshore_life_span

        else:
            # 자가발전 비용
            model += c_sg_y[y] == lambda_CAPEX_PV_y[y] * (capacity_pv_y[y] - capacity_pv_y[y - 1]) + \
                     lambda_OPEX_PV_y[y] * capacity_pv_y[y] + \
                     lambda_CAPEX_onshore_y[y] * (capacity_onshore_y[y] - capacity_onshore_y[y - 1]) + \
                     lambda_OPEX_onshore_y[y] * capacity_onshore_y[y]

            if sum(demand_y_d_h[y, d, h] for d, h in set_d_h) > 0:
                # 기업PPA 비용
                model += c_ppa_y[y] == lp.lpSum(p_ppa_pv_y_d_h[y, d, h] - p_ppa_pv_y_d_h[y - 1, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_pv_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] - p_ppa_onshore_y_d_h[y - 1, d, h] for d, h in set_d_h) * \
                         (lambda_PPA_onshore_y[y] + lambda_nt_c_y[y] + lambda_AS_payment_y[y] + lambda_welfare_y[y]) + \
                         c_ppa_y[y - 1]

            else:
                pass

            # 잔존가치 비용
            model += c_residual_y[y] == lambda_CAPEX_PV_y[y] * (capacity_pv_y[y] - capacity_pv_y[y - 1]) * (pv_life_span - y_tilda) / pv_life_span + \
                     lambda_CAPEX_onshore_y[y] * (capacity_onshore_y[y] - capacity_onshore_y[y - 1]) * (onshore_life_span - y_tilda) / onshore_life_span

        # 기업PPA 망 기본요금
        model += c_ppa_network_basic_y[y] == (lambda_nt_d_y[y] * load_cap * 12) * u_y[y]  # 일단, 발전사업자 최대이용전력과 요금적용전력은 수요자와 같다고 가정.

        # 보완공급
        if sum(demand_y_d_h[y, d, h] for d, h in set_d_h) > 0:
            model += c_tariff_y[y] == lp.lpSum(p_tariff_y_d_h[y, d, h] *
                                               (lambda_tariff_y_d_h[y, d, h] +
                                                lambda_climate_y[y] +
                                                lambda_fuel_adjustment_y[y]) for d, h in set_d_h) + \
                     lambda_tariff_fixed_won_per_kW * load_cap * 12
        else:
            pass

        # 인증서 비용
        model += c_eac_y[y] == p_eac_y[y] * lambda_eac_y[y]

        # 손실정산금
        model += c_loss_payment_y[y] == lambda_loss_payment_y[y] * lp.lpSum(p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h)      # 기업ppa에 할당되는 요금 (한전 평균 전기요금 단가로 계산됨)

        # 전력산업기반기금 (tariff)
        model += c_funding_tariff_y[y] == c_tariff_y[y] * ratio_tariff_funding_y[y]
        
        # 전력산업기반기금 (PPA)
        model += c_funding_ppa_y[y] == (c_ppa_y[y] + c_loss_payment_y[y]) * ratio_ppa_funding_y[y]

        # 부가가치세
        model += c_commission_kepco_y[y] == c_tariff_y[y] * ratio_commission_tariff_ratio_per_won_y[y]
        model += c_commission_ppa_y[y] == c_ppa_y[y] * ratio_commission_ppa_ratio_per_won_y[y]

    for y in set_y:
        if y != options.year0:
            # 자가발전 용량 제약
            model += capacity_pv_y[y] - capacity_pv_y[y - 1] >= 0
            # 자가발전 용량 제약
            model += capacity_onshore_y[y] - capacity_onshore_y[y - 1] >= 0

            # PPA 장기계약 제약
            model += lp.lpSum(p_ppa_pv_y_d_h[y - 1, d, h] for d, h in set_d_h) <= \
                     lp.lpSum(p_ppa_pv_y_d_h[y, d, h] for d, h in set_d_h)
            model += lp.lpSum(p_ppa_onshore_y_d_h[y - 1, d, h] for d, h in set_d_h) <= \
                     lp.lpSum(p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h)

    # 자가발전 설치용량 제약
    for y in set_y:
        model += capacity_pv_y[y] <= cap_max_sg_pv_y[y]
        model += capacity_pv_y[y] >= cap_min_sg_pv_y[y]
        model += capacity_onshore_y[y] <= cap_max_sg_onshore_y[y]
        model += capacity_onshore_y[y] >= cap_min_sg_onshore_y[y]

    # 한전으로부터 수급
    for y, d, h in set_y_d_h:
        model += p_tariff_y_d_h[y, d, h] <= demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]

    # PPA 조달 제약
    for y, d, h in set_y_d_h:
        if demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] > Apv_d_h[d, h] * cap_max_ppa_pv_y[y]:
            model += p_ppa_pv_y_d_h[y, d, h] <= (Apv_d_h[d, h] * cap_max_ppa_pv_y[y]) * u_y[y]
        else:
            model += p_ppa_pv_y_d_h[y, d, h] <= (demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]) * u_y[y]

        if demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] > Aonshore_d_h[d, h] * cap_max_ppa_onshore_y[y]:
            model += p_ppa_onshore_y_d_h[y, d, h] <= (Aonshore_d_h[d, h] * cap_max_ppa_onshore_y[y]) * u_y[y]
        else:
            model += p_ppa_onshore_y_d_h[y, d, h] <= (demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]) * u_y[y]

    # RE100 제약
    if options.customize_achievement_flag:
        for y in set_y:
            model += lp.lpSum(Apv_d_h[d, h] * capacity_pv_y[y] + Aonshore_d_h[d, h] * capacity_onshore_y[y] +
                              p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h) + p_eac_y[y] == \
                     lp.lpSum(demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] for d, h in set_d_h) * (achievement_ratio_y[y] / 100)

    else:
        for y in set_y:
            model += lp.lpSum(Apv_d_h[d, h] * capacity_pv_y[y] + Aonshore_d_h[d, h] * capacity_onshore_y[y] +
                              p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] for d, h in set_d_h) + p_eac_y[y] >= \
                     lp.lpSum(demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h] for d, h in set_d_h) * (achievement_ratio_y[y] / 100)

    # 수요공급
    for y, d, h in set_y_d_h:
        model += Apv_d_h[d, h] * capacity_pv_y[y] + Aonshore_d_h[d, h] * capacity_onshore_y[y] + \
                 p_tariff_y_d_h[y, d, h] + p_ppa_pv_y_d_h[y, d, h] + p_ppa_onshore_y_d_h[y, d, h] >= \
                 demand_y_d_h[y, d, h] - ee_y_d_h[y, d, h]

    print('RE100 simulator, start solve -> solve status:', end=' ')
    if solver_name == 'CBC':
        solver = lp.PULP_CBC_CMD(gapRel=gaprel, msg=options.solver_message)
    elif solver_name == 'gurobi':
        solver = lp.GUROBI_CMD(gapRel=gaprel, msg=options.solver_message)
    elif solver_name == 'cplex':
        solver = lp.CPLEX_CMD(gapRel=gaprel, msg=options.solver_message)
    else:
        print('solver selection error')

    model.solve(solver)
    print(lp.LpStatus[model.status])

    # 결과저장
    result = dict()

    result['lambda_CAPEX_PV_y'] = lambda_CAPEX_PV_y
    result['lambda_OPEX_PV_y'] = lambda_OPEX_PV_y
    result['lambda_CAPEX_onshore_y'] = lambda_CAPEX_onshore_y
    result['lambda_OPEX_onshore_y'] = lambda_OPEX_onshore_y

    result['lambda_eac_y'] = lambda_eac_y
    result['lambda_PPA_pv_y'] = lambda_PPA_pv_y
    result['lambda_PPA_onshore_y'] = lambda_PPA_onshore_y
    result['lambda_AS_payment_y'] = lambda_AS_payment_y
    result['lambda_climate_y'] = lambda_climate_y
    result['lambda_nt_c_y'] = lambda_nt_c_y
    result['lambda_nt_d_y'] = lambda_nt_d_y
    result['lambda_fuel_adjustment_y'] = lambda_fuel_adjustment_y
    result['lambda_loss_payment_y'] = lambda_loss_payment_y
    result['lambda_welfare_y'] = lambda_welfare_y

    result['rate_pv'] = input_parameters_pulp.rate_pv
    result['rate_onshore'] = input_parameters_pulp.rate_onshore
    result['tariff_y'] = input_parameters_pulp.tariff_y

    result['npv'] = model.objective.value()

    lambda_tariff_y_d_h_dict = dict()

    capacity_pv_y_dict = dict()
    capacity_onshore_y_dict = dict()

    c_sg_y_dict = dict()
    c_tariff_y_dict = dict()
    c_ppa_y_dict = dict()
    c_eac_y_dict = dict()
    c_loss_payment_y_dict = dict()
    c_funding_tariff_y_dict = dict()
    c_funding_ppa_y_dict = dict()
    c_commission_kepco_y_dict = dict()
    c_commission_ppa_y_dict = dict()
    c_residual_y_dict = dict()
    c_ppa_network_basic_y_dict = dict()

    p_eac_y_dict = dict()
    p_sg_pv_y_d_h_dict = dict()
    p_sg_onshore_y_d_h_dict = dict()
    p_tariff_y_d_h_dict = dict()
    p_ppa_pv_y_d_h_dict = dict()
    p_ppa_onshore_y_d_h_dict = dict()

    for y in set_y:
        p_eac_y_dict[y] = p_eac_y[y].value()
        capacity_pv_y_dict[y] = capacity_pv_y[y].value()
        capacity_onshore_y_dict[y] = capacity_onshore_y[y].value()
        c_sg_y_dict[y] = c_sg_y[y].value()
        c_tariff_y_dict[y] = c_tariff_y[y].value()
        c_ppa_y_dict[y] = c_ppa_y[y].value()
        c_eac_y_dict[y] = c_eac_y[y].value()
        c_residual_y_dict[y] = c_residual_y[y].value()
        c_loss_payment_y_dict[y] = c_loss_payment_y[y].value()
        c_funding_tariff_y_dict[y] = c_funding_tariff_y[y].value()
        c_funding_ppa_y_dict[y] = c_funding_ppa_y[y].value()
        c_commission_kepco_y_dict[y] = c_commission_kepco_y[y].value()
        c_commission_ppa_y_dict[y] = c_commission_ppa_y[y].value()
        c_ppa_network_basic_y_dict[y] = c_ppa_network_basic_y[y].value()

        p_sg_pv_y_d_h_dict[y] = dict()
        p_sg_onshore_y_d_h_dict[y] = dict()
        p_tariff_y_d_h_dict[y] = dict()
        p_ppa_pv_y_d_h_dict[y] = dict()
        p_ppa_onshore_y_d_h_dict[y] = dict()

        lambda_tariff_y_d_h_dict[y] = dict()

        for d in set_d:
            p_sg_pv_y_d_h_dict[y][d] = dict()
            p_sg_onshore_y_d_h_dict[y][d] = dict()
            p_tariff_y_d_h_dict[y][d] = dict()
            p_ppa_pv_y_d_h_dict[y][d] = dict()
            p_ppa_onshore_y_d_h_dict[y][d] = dict()

            lambda_tariff_y_d_h_dict[y][d] = dict()

            for h in set_h:
                p_sg_pv_y_d_h_dict[y][d][h] = Apv_d_h[d, h] * capacity_pv_y[y].value()
                p_sg_onshore_y_d_h_dict[y][d][h] = Aonshore_d_h[d, h] * capacity_onshore_y[y].value()
                p_tariff_y_d_h_dict[y][d][h] = p_tariff_y_d_h[y, d, h].value()
                p_ppa_pv_y_d_h_dict[y][d][h] = p_ppa_pv_y_d_h[y, d, h].value()
                p_ppa_onshore_y_d_h_dict[y][d][h] = p_ppa_onshore_y_d_h[y, d, h].value()

                lambda_tariff_y_d_h_dict[y][d][h] = lambda_tariff_y_d_h[y, d, h]

    result['p_sg_pv_y_d_h'] = p_sg_pv_y_d_h_dict
    result['p_sg_onshore_y_d_h'] = p_sg_onshore_y_d_h_dict
    result['p_tariff_y_d_h'] = p_tariff_y_d_h_dict
    result['p_ppa_pv_y_d_h'] = p_ppa_pv_y_d_h_dict
    result['p_ppa_onshore_y_d_h'] = p_ppa_onshore_y_d_h_dict
    result['p_eac_y'] = p_eac_y_dict
    result['capacity_pv_y'] = capacity_pv_y_dict
    result['capacity_onshore_y'] = capacity_onshore_y_dict

    result['c_sg_y'] = c_sg_y_dict
    result['c_tariff_y'] = c_tariff_y_dict
    result['c_ppa_y'] = c_ppa_y_dict
    result['c_eac_y'] = c_eac_y_dict
    result['c_residual_y'] = c_residual_y_dict
    result['c_loss_payment_y'] = c_loss_payment_y_dict
    result['c_funding_tariff_y'] = c_funding_tariff_y_dict
    result['c_funding_ppa_y'] = c_funding_ppa_y_dict

    result['c_commission_kepco_y'] = c_commission_kepco_y_dict
    result['c_commission_ppa_y'] = c_commission_ppa_y_dict
    result['c_ppa_network_basic_y'] = c_ppa_network_basic_y_dict
    result['lambda_tariff_y_d_h'] = lambda_tariff_y_d_h_dict

    return result