import os.path

import pandas as pd
import numpy as np
import random
import copy
from sklearn.linear_model import LinearRegression


class ProbOptions:
    model_basis_year = 2021
    present_year = 2022

    def __init__(self):
        self.solver_message = False
        self.load_distribution_high_voltage_using_flag = False
        self.gen_transmission_using_flag = False

        self.run_simulation_flag = False
        self.run_simulation_by_case_flag = False  # True: 미리 설정한 case별 분석 수행, False: 단일 분석 수행
        self.parallel_processing_flag = False  # 병렬처리

        self.complementary_tariff_table_origin_flag = True  # 보완공급 버전 여부   True: 기존 보완공급요금제, False: 평균으로 변환한 요금제

        self.customize_achievement_flag = False  # True: RE100 달성 비중을 무조건적으로 맞추기, False: 자유롭게 맞추기(단, 최소값은 유지)
        self.customize_self_generation_flag = True  # True: 자가발전 용량을 전력소비량의 0% 기준, 1% 기준으로 설치, False: 자유롭게 자가발전 용량 결정
        self.customize_energy_efficiency_flag = True  # True: 에너지효율화 미고려 False: 고려

        self.customize_exemption_flag = False  # True: 망이용, 부가정산, 송배전손실 등 PPA로 인한 부가비용 면제, False: 부가비용 고려
        self.customize_setup_NetworkTariff = True  # True: 망요금을 강제로 설정, False: 송배전이용요금을 준수하여 결정

        self.result_analysis_flag = False
        self.plot_optimal_portfolio_flag = True
        self.plot_yearly_cost_flag = True

        self.excel_optimal_portfolio_flag = True  # 포트폴리오 구성비중 엑셀
        self.excel_optimal_portfolio_cost_flag = True  # 포트폴리오 구성비용 엑셀
        self.excel_optimal_portfolio_spec_flag = True  # 포트폴리오 구성스펙 엑셀
        self.excel_average_cost_flag = True  # 연간 평균비용 엑셀
        self.excel_unit_price_by_year_flag = True
        self.excel_unit_price_by_items_flag = True
        self.excel_options_unit_price_by_year_flag = True
        self.excel_ppa_capacity = True

        self.result_BAU_analysis_flag = False  #
        self.excel_yearly_cost_in_BAUscen_flag = True  # 연간 평균비용 엑셀
        self.plot_yearly_cost_in_BAUscen_flag = True  # 연간 평균비용 Plot
        self.excel_unit_price_by_year_in_BAUscen_flag = True

        self.result_integration_analysis_flag = False

        self.etc_plot_demand_pattern_flag = False  # 업종별 전력 수요패턴 그리기

        self.loc_pv_data = f"data/전북_한국서부_군산복합2단계_2019_태양광.xlsx"
        self.loc_onshore_data = f"data/전남_한국서부_화순_2019_풍력.xlsx"
        self.loc_tariff_table = f"data/tariff_table.xlsx"
        self.loc_tariff_avg_2020 = f"data/tariff.xlsx"
        self.loc_tariff_rate = f"data/tariff_rate.xlsx"
        self.loc_rec_avg = f"data/rec.xlsx"
        self.loc_rec_rate = f"data/rec_rate.xlsx"
        self.loc_ppa_avg = f"data/ppa.xlsx"
        self.loc_lcoe_rate = f"data/lcoe_rate.xlsx"
        self.loc_inst_cost_avg = f"data/re_inst_cost.xlsx"
        self.loc_as_payment_avg = f"data/as_payment.xlsx"
        self.loc_cap_max_ppa = f"data/cap_max_ppa.xlsx"
        self.loc_cap_max_sg = f"data/cap_max_sg.xlsx"
        self.loc_ee_cost = f"data/energy efficiency_cost.xlsx"
        self.loc_dema_raw1 = f"data/업종별 에너지 사용 패턴 데이터_200831.xlsx"
        self.loc_dema_raw2 = f"data/AMR고객별부하현황_세부업종별_2019.csv"
        self.loc_dema_info = f"data/업종별 에너지 사용 패턴 데이터_200831.xlsx"
        self.loc_eff = f"data/energy efficiency.xlsx"

        self.customize_self_generation_ratio = 1  # %

        self.loc_result = f"D:/RE100_NEXTgroup/result"
        self.date_result = "230126"
        self.set_result_loc(opt='부가비용')
        self.loc_plot = f"D:/RE100_NEXTgroup/plot"
        self.loc_excel = f"D:/RE100_NEXTgroup/excel"

        self.year0 = 2022
        self.year1 = 2040

        self.load_cap = 100000  # 기업 최대 소비전력 kW
        self.gaprel = 0.001
        self.contract_voltage = 154
        self.tariff_ppa = '기업PPA,고압B' if self.contract_voltage == 154 else '기업PPA,고압C' if self.contract_voltage >= 345 else '기업PPA,고압A'
        self.tariff_bau = '을,고압B' if self.contract_voltage == 154 else '을,고압C' if self.contract_voltage >= 345 else '을,고압A'

        self.unit = 10 ** 8

        achievement_ratio = dict()
        achievement_ratio.update(self.cal_ratio_by_trend('linear', 0, 0, 2022, 2024))
        achievement_ratio.update(self.cal_ratio_by_trend('linear', 18.55, 0, 2024, 2029))
        achievement_ratio.update(self.cal_ratio_by_trend('linear', 60, 18.55, 2029, 2035))
        achievement_ratio.update(self.cal_ratio_by_trend('linear', 100, 60, 2035, 2040))
        self.achievement_ratio_y = achievement_ratio

        self.country = 'KR'
        """
        발전지역별 송전이용요금단가
        * 발전 배전저압, 배전고압 비용 X (PPA 송배전망 이용요금 산정 계산기 참고함_솔라커넥트)
        1: 수도권 북부지역
        2: 수도권 남부지역
        3: 비수도권 지역
        4: 제주지역
        """
        self.region_gen = 3
        """
        수요지역별 송전이용요금단가
        1: 수도권지역
        2: 비수도권지역
        3: 제주지역
        """
        self.region_load = 2

        self.scenario_num = 1
        self.pp_num = 1000

        self.self_generator_set = [
            'photovoltaic',
            'onshore wind'
        ]

    def set_result_loc(self, opt):
        opt_name = self.set_pp_name(opt)
        self.set_result_dir(opt_name, self.loc_result, self.date_result)
        self.loc_pp_result = f"{self.loc_result}/{self.date_result}/{opt_name}"

    def set_result_dir(self, opt_name, loc_result, date_result):
        if os.path.isdir(f"{loc_result}/{date_result}"):
            pass
        else:
            os.makedirs(f"{loc_result}/{date_result}")
        if os.path.isdir(f"{loc_result}/{date_result}/{opt_name}"):
            pass
        else:
            os.makedirs(f"{loc_result}/{date_result}/{opt_name}")

    def set_pp_name(self, opt):
        if opt == '부가비용':
            if self.customize_exemption_flag:
                name = '부가비용면제'
            else:
                name = '부가비용고려'
        return name

    def cal_ratio_by_trend(self, trend_type, target_rate, init_rate, y0, y1):
        set_dict = dict()
        slope = (target_rate - init_rate) / (y1 - y0)
        intercept = init_rate - slope * y0
        if trend_type == 'constant':
            for y in range(y0, y1 + 1):
                set_dict[y] = target_rate

        elif trend_type == 'linear':
            for y in range(y0, y1 + 1):
                set_dict[y] = np.round(slope * y + intercept, 2)
        return set_dict


class ReadInputData:
    def __init__(self, options: ProbOptions):
        print("reading input data")
        Apv_raw = pd.read_excel(f"{options.loc_pv_data}", sheet_name=None)
        Apv_raw = Apv_raw['Sheet1'].loc[:, 1:24] / Apv_raw['Sheet2'].iloc[0].iloc[0]
        self.Apv_raw = Apv_raw.values

        Aonshore_raw = pd.read_excel(f"{options.loc_onshore_data}", sheet_name=None)
        Aonshore_raw = Aonshore_raw['Sheet1'].loc[:, 1:24] / Aonshore_raw['Sheet2'].iloc[0].iloc[0]
        self.Aonshore_raw = Aonshore_raw.values

        self.tariff_table_ppa = pd.read_excel(f"{options.loc_tariff_table}", sheet_name=options.tariff_ppa, index_col=0)
        self.tariff_avg_2020 = pd.read_excel(f"{options.loc_tariff_avg_2020}", sheet_name=options.country)
        self.tariff_rate = pd.read_excel(f"{options.loc_tariff_rate}", sheet_name=options.country)
        self.rec_avg = pd.read_excel(f"{options.loc_rec_avg}", sheet_name=options.country)
        self.rec_rate = pd.read_excel(f"{options.loc_rec_rate}", sheet_name=options.country)
        self.ppa_avg = pd.read_excel(f"{options.loc_ppa_avg}", sheet_name=options.country)
        self.lcoe_rate = pd.read_excel(f"{options.loc_lcoe_rate}", sheet_name=options.country)
        self.inst_cost_avg = pd.read_excel(f"{options.loc_inst_cost_avg}", sheet_name=options.country)
        self.as_payment_avg = pd.read_excel(f"{options.loc_as_payment_avg}", sheet_name=options.country, index_col=0)
        self.cap_max_ppa = pd.read_excel(f"{options.loc_cap_max_ppa}", sheet_name='Sheet1', index_col=0, header=0)
        self.cap_max_sg = pd.read_excel(f"{options.loc_cap_max_sg}", sheet_name='Sheet1', index_col=0, header=0)
        self.ee_cost = pd.read_excel(f"{options.loc_ee_cost}", sheet_name='Sheet1', index_col=0, header=0)


        # customized
        dema_fac_dict = dict()
        dema_fac_dict['자동차 부품'] = \
        pd.read_excel(f"{options.loc_dema_raw1}", sheet_name='자동차부품', index_col=0, header=0)[
            'Power consumption'].values.reshape(365, 24)
        dema_fac_dict['일반기계 등'] = \
        pd.read_excel(f"{options.loc_dema_raw1}", sheet_name='일반기계', index_col=0, header=0)[
            'Power consumption'].values.reshape(365, 24)
        dema_fac_dict['SK데이터센터'] = \
        pd.read_excel(f"{options.loc_dema_raw1}", sheet_name='데이터센터', index_col=0, header=0)[
            'Power consumption'].values.reshape(365, 24)

        dema_raw2_df = pd.read_csv(f"{options.loc_dema_raw2}", encoding='euc-kr')
        dema_fac_dict['OCI'] = dema_raw2_df[dema_raw2_df['업종'].str.contains('화학물질')].iloc[:, 3:].values
        dema_fac_dict['수소생산클러스터'] = dema_raw2_df[dema_raw2_df['업종'].str.contains('전기, 가스')].iloc[:, 3:].values

        dema_info = pd.read_excel(f"{options.loc_dema_info}",
                                    sheet_name='업종별 수요정보',
                                    index_col=0,
                                    header=0).loc[['SK데이터센터', 'OCI', '자동차 부품', '일반기계 등', '수소생산클러스터'],
        range(options.year0, options.year1 + 1)]

        self.demand_factor = dema_fac_dict
        self.demand_info = dema_info
        self.eff = pd.read_excel(f"{options.loc_eff}", sheet_name='Sheet1', index_col=0,
                                     header=0)


class ParameterPulpFrom:

    def __init__(self, options: ProbOptions, IFN: ReadInputData):
        self.set_y = np.arange(options.year0, options.year1 + 1, 1).astype('int').tolist()
        self.set_d = np.arange(1, 365 + 1, 1).astype('int').tolist()
        self.set_h = np.arange(1, 24 + 1, 1).astype('int').tolist()

        ## next 자문 버전 ##
        self.discount_rate = 0.045
        self.pv_life_span = 20
        self.onshore_life_span = 20

        self.demand_y_d_h = self.make_demand_pattern(IFN, options)

        if options.customize_energy_efficiency_flag:
            self.lambda_ee_y = self.none_to_dict_y(0)
            self.ee_y_d_h = self.none_to_dict_y_d_h(0)
        else:
            self.lambda_ee_y = self.df_to_dict_y(IFN.ee_cost, '비용(원)')
            self.ee_y_d_h = self.make_ee_pattern(IFN, options)

        if options.customize_self_generation_flag:
            self.cap_max_sg_pv_y = self.customize_sg_cap(options, IFN.Apv_raw, self.demand_y_d_h, self.ee_y_d_h)
            self.cap_min_sg_pv_y = self.customize_sg_cap(options, IFN.Apv_raw, self.demand_y_d_h, self.ee_y_d_h)
            self.cap_max_sg_onshore_y = self.none_to_dict_y(0)  # 일단, 자가발전으로 육상풍력  미고려
            self.cap_min_sg_onshore_y = self.none_to_dict_y(0)

        else:
            self.cap_max_sg_pv_y = self.df_to_dict_y(IFN.cap_max_sg, 'pv')  # 산단 부지 내 설치 가능한 태양광 자가발전 최대용량
            self.cap_min_sg_pv_y = self.none_to_dict_y(0)
            self.cap_max_sg_onshore_y = self.df_to_dict_y(IFN.cap_max_sg, 'onshore')  # 산단 부지 내 설치 가능한 육상풍력 자가발전 최대용량
            self.cap_min_sg_onshore_y = self.none_to_dict_y(0)

        self.cap_max_ppa_pv_y = self.df_to_dict_y(IFN.cap_max_ppa, 'pv')  # 기업PPA with PV 계약 가능한 용량
        self.cap_max_ppa_onshore_y = self.df_to_dict_y(IFN.cap_max_ppa, 'onshore')  # 기업PPA with onshore 계약 가능한 용량

        self.ratio_commission_tariff_ratio_per_won_y = self.fill_value_to_dict(
            0.1)  # 부가가치세, 출처: https://cyber.kepco.co.kr/ckepco/front/jsp/CY/J/A/CYJAPP000NFL.jsp
        self.ratio_tariff_funding_y = self.fill_value_to_dict(0.037)  # 전력산업기반기금 ratio. 모든 항목의 합 * ratio. 출처: 한전

        self.rate_min_pv = IFN.lcoe_rate[IFN.lcoe_rate['type'] == 'pv0']['rate'].iloc[0]
        self.rate_max_pv = IFN.lcoe_rate[IFN.lcoe_rate['type'] == 'pv1']['rate'].iloc[0]
        self.rate_min_onshore = IFN.lcoe_rate[IFN.lcoe_rate['type'] == 'onshore0']['rate'].iloc[0]
        self.rate_max_onshore = IFN.lcoe_rate[IFN.lcoe_rate['type'] == 'onshore1']['rate'].iloc[0]

        self.rate_max_tariff = self.cal_value_using_regression(
            data=IFN.tariff_rate[IFN.tariff_rate['year'] >= 2015]['USD/kWh'].values.reshape(-1, 1),
            y=options.year1 - options.model_basis_year + 1, init=IFN.tariff_rate['USD/kWh'].values.reshape(-1, 1)[
                -1])  # year1까지 평균 전기요금 증가율 도출 (과거 데이터 기반(2015년 이상) 선형추세선 적용)
        self.rate_min_tariff = 1

        self.rate_pv = self.cal_random_ratio(self.rate_min_pv, self.rate_max_pv, 100)
        self.rate_onshore = self.cal_random_ratio(self.rate_min_onshore, self.rate_max_onshore, 100)
        self.rate_tariff = self.cal_random_ratio(self.rate_min_tariff, self.rate_max_tariff, 100)

        self.tariff_average = IFN.tariff_avg_2020['won/kWh'].iloc[0]  # 2020년 평균 산업용 전기요금 단가, 환율: 1342.81원/$
        self.tariff_y = self.cal_trend(options, 'linear', self.tariff_average, self.rate_tariff, 1)
        self.lambda_tariff_ppa_y_d_h = self.cal_tariff_table(options, self.tariff_y, self.tariff_average, IFN.tariff_table_ppa)  # 보완공급
        self.lambda_tariff_fixed_won_per_kW = IFN.tariff_table_ppa.loc['기본요금', :].iloc[0]

        self.p_pv_install = IFN.inst_cost_avg[IFN.inst_cost_avg['type'] == 'pv']['won/kw'].iloc[0]
        self.lambda_CAPEX_PV_y = self.cal_trend(options, 'linear', self.p_pv_install, self.rate_pv, 1)  # 출처: IRENA (엑셀참고)
        self.lambda_OPEX_PV_y = self.cal_trend(options, 'linear', self.p_pv_install * 0.02, self.rate_pv, 1)  # CAPEX의 3% 가정

        self.p_onshore_install = IFN.inst_cost_avg[IFN.inst_cost_avg['type'] == 'onshore']['won/kw'].iloc[0]
        self.lambda_CAPEX_onshore_y = self.cal_trend(options, 'linear', self.p_onshore_install, self.rate_onshore, 1)  # 출처: IRENA (엑셀참고)
        self.lambda_OPEX_onshore_y = self.cal_trend(options, 'linear', self.p_onshore_install * 0.02, self.rate_onshore, 1)  # CAPEX의 3% 가정

        self.lambda_PPA_pv_y = self.cal_trend(options=options,
                                              trend_type='linear',
                                              init_value=IFN.ppa_avg[IFN.ppa_avg['source'] == 'pv']['won'].iloc[0],
                                              target_rate=self.rate_pv,
                                              init_rate=1)

        self.lambda_PPA_onshore_y = self.cal_trend(options=options,
                                                   trend_type='linear',
                                                   init_value=
                                                   IFN.ppa_avg[IFN.ppa_avg['source'] == 'onshore']['won'].iloc[0],
                                                   target_rate=self.rate_onshore,
                                                   init_rate=1)

        self.lambda_eac_y = self.make_rec_set(options, IFN.rec_rate['rate'].values,
                                              IFN.rec_avg['won/kWh'].iloc[0])

        # 22.10.11 신규 항목 추가 (복지및특례비용, 기후환경요금, 연료비조정액)
        self.lambda_welfare_y = self.fill_value_to_dict(0)  # 22.10.11 한전 엔터 에너지마켓플레이스 참고, 22.12.01 복지 및 특례할인은 제3자 PPA만 부과
        self.lambda_climate_y = self.fill_value_to_dict(9)  # 공급약관 및 세칙 개정 사항 안내('22.12.30) 참고
        self.lambda_fuel_adjustment_y = self.fill_value_to_dict(5)  # 22.10.11 한전 전기요금계산기 참고

        self.Apv_d_h = self.np_to_dict_d_h(IFN.Apv_raw)
        self.Aonshore_d_h = self.np_to_dict_d_h(IFN.Aonshore_raw)

        self.lambda_nt_d, self.lambda_nt_c = self.cal_network_tariff(options)  # nt_d: won/kW, nt_c: won/kWh. 출처: 한전송배전망이용요금표
        self.lambda_AS_payment = IFN.as_payment_avg.loc['AS payment', 'won/kWh']
        self.lambda_loss_payment = self.cal_loss_payment(options,
                                                         self.tariff_average)  # 손실정산금 단가 won/kWh. 손실율 * 평균전기요금단가(2020 산업용 전기요금 단가)
        self.ratio_ppa_funding_y = self.fill_value_to_dict(0.037)  # 전력산업기반기금 ratio. 모든 항목의 합 * ratio. 출처: 한전
        self.ratio_commission_ppa_ratio_per_won_y = self.fill_value_to_dict(0.1)

        self.lambda_loss_payment_y = self.fill_value_to_dict(self.lambda_loss_payment)
        self.lambda_AS_payment_y = self.fill_value_to_dict(self.lambda_AS_payment)
        self.lambda_nt_d_y = self.fill_value_to_dict(self.lambda_nt_d)
        self.lambda_nt_c_y = self.fill_value_to_dict(self.lambda_nt_c)

    def fill_value_to_dict(self, value):
        set_y = self.set_y
        set_dict = dict()
        for y in set_y:
            set_dict[y] = value
        return set_dict

    def customize_sg_cap(self, options, A_file, demand_y_d_h, ee_y_d_h):
        set_y = self.set_y
        set_d = self.set_d
        set_h = self.set_h
        cap_sg_y = dict()

        for y in set_y:
            demand = 0
            ee = 0
            for d in set_d:
                for h in set_h:
                    demand += demand_y_d_h[y, d, h]
                    ee += ee_y_d_h[y, d, h]
            sg_gen = demand * options.customize_self_generation_ratio / 100
            achienve_gen = (demand - ee) * options.achievement_ratio_y[y] / 100
            if sg_gen > achienve_gen:
                cap_sg_y[y] = achienve_gen / np.sum(A_file)
            else:
                cap_sg_y[y] = sg_gen / np.sum(A_file)
        return cap_sg_y

    def make_pattern(self, options, factor, amount_list):
        set_y = self.set_y

        frame = np.zeros((options.year1 - options.year0 + 1, 365, 24))  # 윤년 제외

        index_1 = 0
        for y in set_y:
            frame[index_1, :, :] = factor * (amount_list[y] / np.sum(factor))
            index_1 += 1

        return frame

    def make_ee_pattern(self, IFN, options):
        set_y = self.set_y
        set_d = self.set_d
        set_h = self.set_h

        ee_factor = np.ones((365, 24))
        ee = self.make_pattern(options, ee_factor, IFN.eff.loc['총합', :]) * 1000  # kWh

        ee_y_d_h = dict()
        for y in set_y:
            for d in set_d:
                for h in set_h:
                    ee_y_d_h[y, d, h] = ee[y - options.year0, d - 1, h - 1]

        return ee_y_d_h

    def make_demand_pattern(self, IFN, options):
        set_y = self.set_y
        set_d = self.set_d
        set_h = self.set_h

        demand_factors = copy.deepcopy(IFN.demand_factor)
        for k in IFN.demand_factor.keys():
            demand_factors[k] = demand_factors[k] / np.max(demand_factors[k])

        demand_y_d_h_arr = np.zeros((options.year1 - options.year0 + 1, 365, 24))
        for k in demand_factors.keys():
            demand_y_d_h_arr += self.make_pattern(options, demand_factors[k], IFN.demand_info.loc[k, :]) * 1000

        demand_y_d_h = dict()
        for y in set_y:
            for d in set_d:
                for h in set_h:
                    demand_y_d_h[y, d, h] = demand_y_d_h_arr[y - options.year0, d - 1, h - 1]

        return demand_y_d_h

    def none_to_dict_y_d_h(self, value):
        set_y = self.set_y
        set_d = self.set_d
        set_h = self.set_h
        set_dict = dict()
        for y in set_y:
            for d in set_d:
                for h in set_h:
                    set_dict[y, d, h] = value
        return set_dict

    def none_to_dict_y(self, value):
        set_y = self.set_y

        set_dict = dict()
        for y in set_y:
            set_dict[y] = value
        return set_dict

    def df_to_dict_y(self, file, item):
        set_y = self.set_y

        set_dict = dict()
        for y in set_y:
            set_dict[y] = file.loc[item, y]
        return set_dict

    def np_to_dict_d_h(self, np_file):
        set_d = self.set_d
        set_h = self.set_h

        set_dict = dict()
        for d in set_d:
            for h in set_h:
                set_dict[d, h] = float(np_file[d - 1, h - 1])

        return set_dict

    def cal_value_using_regression(self, data, y, init):
        line_fitter = LinearRegression()
        x_data = np.arange(1, data.shape[0] + 1, 1).reshape(-1, 1)
        line_fitter.fit(X=x_data, y=data)

        x_predict = np.arange(x_data[-1][0] + 1, x_data[-1][0] + y + 1).reshape(-1, 1)
        y_predict = line_fitter.coef_ * x_predict + line_fitter.intercept_
        R = (y_predict / init).reshape(-1)
        return R[-1]

    def cal_loss_payment(self, options, tariff):
        if options.gen_transmission_using_flag and options.load_distribution_high_voltage_using_flag == False:
            loss = 1.57
        elif options.gen_transmission_using_flag and options.load_distribution_high_voltage_using_flag:
            loss = 3.54
        elif options.gen_transmission_using_flag == False and options.load_distribution_high_voltage_using_flag == False:
            loss = 1.99
        return loss / 100 * tariff

    def cal_network_tariff(self, options):
        if options.customize_setup_NetworkTariff:
            nt_d = 0
            nt_c = 14.20
        else:
            nt_d = 0
            nt_c = 0
            # 수용가 배전고압 수전
            if options.load_distribution_high_voltage_using_flag:
                nt_d += 548
                nt_c += 3.05
            # 수용가 송전 수전
            else:
                if options.region_load == 1:
                    nt_c += 2.44
                elif options.region_load == 2:
                    nt_c += 1.42
                elif options.region_load == 3:
                    nt_c += 6.95
                else:
                    print('error: 수요지역 input error')

                nt_d += 667.61

            if options.gen_transmission_using_flag:
                nt_d += 667.36
                if options.region_gen == 1:
                    nt_c += 1.25
                elif options.region_gen == 2:
                    nt_c += 1.20
                elif options.region_gen == 3:
                    nt_c += 1.92
                elif options.region_gen == 4:
                    nt_c += 1.90
                else:
                    print('error: 발전지역 input error')
        return nt_d, nt_c

    def cal_random_ratio(self, r0, r1, per):
        new_r0 = r0 + (r1 - r0) * (1 - per / 100) / 2
        new_r1 = r1 - (r1 - r0) * (1 - per / 100) / 2

        return np.random.uniform(new_r0, new_r1)

    def cal_achievement_rate(self, options):
        set_y = self.set_y

        set_dict = dict()
        count = 1
        for y in set_y:
            if options.achievement_trend == 'constant':
                set_dict[y] = self.init_achievement
            elif options.achievement_trend == 'linear':
                set_dict[y] = self.init_achievement * count
                count += 1
        return set_dict

    def cal_trend(self, options, trend_type, init_value, target_rate, init_rate):
        set_y = self.set_y

        set_dict = dict()
        slope = (target_rate - init_rate) / (options.year1 - options.year0)
        intercept = init_rate - slope * options.year0

        for y in set_y:
            if trend_type == 'linear':
                set_dict[y] = float(init_value * (slope * y + intercept))
            elif trend_type == 'constant':
                set_dict[y] = float(init_value)

        return set_dict

    # 공휴일 또는 주말 요금 반영 안함
    def cal_tariff_table(self, options, tariff_y, tariff_average, tariff_table):
        set_y = self.set_y
        set_d = self.set_d
        set_h = self.set_h

        if options.complementary_tariff_table_origin_flag:
            table = pd.DataFrame(np.zeros((365, 24)), columns=np.arange(0, 24, 1))

            summer = np.arange(151, 242 + 1, 1)
            winter = np.arange(0, 58 + 1, 1)
            winter = np.append(winter, np.arange(304, 364 + 1, 1))
            others = np.arange(59, 150 + 1, 1)
            others = np.append(others, np.arange(243, 303 + 1, 1))
            ##
            table.loc[summer, [23, 24, 1, 2, 3, 4, 5, 6, 7, 8]] = tariff_table.loc['경부하', '여름철']
            table.loc[summer, [9, 10, 11, 13, 19, 20, 21, 22]] = tariff_table.loc['중간부하', '여름철']
            table.loc[summer, [12, 14, 15, 16, 17, 18]] = tariff_table.loc['최대부하', '여름철']
            table.loc[winter, [23, 24, 1, 2, 3, 4, 5, 6, 7, 8]] = tariff_table.loc['경부하', '겨울철']
            table.loc[winter, [9, 13, 14, 15, 16, 20, 21, 22]] = tariff_table.loc['중간부하', '겨울철']
            table.loc[winter, [10, 11, 12, 17, 18, 19]] = tariff_table.loc['최대부하', '겨울철']
            table.loc[others, [23, 24, 1, 2, 3, 4, 5, 6, 7, 8]] = tariff_table.loc['경부하', '봄가을철']
            table.loc[others, [9, 10, 11, 13, 19, 20, 21, 22]] = tariff_table.loc['중간부하', '봄가을철']
            table.loc[others, [12, 14, 15, 16, 17, 18]] = tariff_table.loc['최대부하', '봄가을철']

        # 모든시간대에 대한 평균요금제 적용
        else:
            table = pd.DataFrame(np.ones((365, 24)) * tariff_table.loc[:, '여름철':'겨울철'].values.mean(),
                                 columns=np.arange(0, 24, 1))

        axis_new = 0
        for y in set_y:
            rate = tariff_y[y] / tariff_average
            if axis_new == 0:
                table = np.expand_dims(table, axis=0)
            else:
                table = np.append(table, np.expand_dims(table[0, :, :] * rate, axis=0), axis=0)

            axis_new += 1

        table_y_d_h = dict()
        for y in set_y:
            for d in set_d:
                for h in set_h:
                    table_y_d_h[y, d, h] = table[y - options.year0, d - 1, h - 1]
        return table_y_d_h

    def make_rec_set(self, options, rate_set, init):
        set_y = self.set_y

        set_dict = dict()
        rec_min = rate_set.min()
        rec_max = rate_set.max()
        for y in set_y:
            if y == options.year0:
                set_dict[y] = float(init)
            else:
                ran_val = np.random.uniform(rec_min, rec_max)
                set_dict[y] = float(set_dict[y - 1] * ran_val)
        return set_dict

