import os.path

import pandas as pd
import numpy as np
import random
import copy
import json
from sklearn.linear_model import LinearRegression


class ProbOptions:
    def __init__(self):
        self.solver_message = True
        # self.load_distribution_high_voltage_using_flag = False
        # self.gen_transmission_using_flag = False
        #
        # self.run_simulation_flag = False
        # self.run_simulation_by_case_flag = False  # True: 미리 설정한 case별 분석 수행, False: 단일 분석 수행
        # self.parallel_processing_flag = False  # 병렬처리
        #
        # self.customize_achievement_flag = False  # True: RE100 달성 비중을 무조건적으로 맞추기, False: 자유롭게 맞추기(단, 최소값은 유지)
        # self.customize_self_generation_flag = False  # True: 자가발전 용량을 전력소비량의 0% 기준, 1% 기준으로 설치, False: 자유롭게 자가발전 용량 결정
        #
        # self.customize_exemption_flag = True  # True: 망이용, 부가정산, 송배전손실 등 PPA로 인한 부가비용 면제, False: 부가비용 고려
        # self.customize_setup_NetworkTariff = True  # True: 망요금을 강제로 설정, False: 송배전이용요금을 준수하여 결정
        #
        # self.result_analysis_flag = False
        # self.plot_optimal_portfolio_flag = True
        # self.plot_yearly_cost_flag = True
        #
        # self.excel_optimal_portfolio_flag = True  # 포트폴리오 구성비중 엑셀
        # self.excel_optimal_portfolio_cost_flag = True  # 포트폴리오 구성비용 엑셀
        # self.excel_optimal_portfolio_spec_flag = True  # 포트폴리오 구성스펙 엑셀
        # self.excel_average_cost_flag = True  # 연간 평균비용 엑셀
        # self.excel_unit_price_by_year_flag = True
        # self.excel_unit_price_by_items_flag = True
        # self.excel_options_unit_price_by_year_flag = True
        # self.excel_ppa_capacity = True
        #
        # self.result_BAU_analysis_flag = False  #
        # self.excel_yearly_cost_in_BAUscen_flag = True  # 연간 평균비용 엑셀
        # self.plot_yearly_cost_in_BAUscen_flag = True  # 연간 평균비용 Plot
        # self.excel_unit_price_by_year_in_BAUscen_flag = True
        #
        # self.result_integration_analysis_flag = False
        #
        # self.etc_plot_demand_pattern_flag = False  # 업종별 전력 수요패턴 그리기



        # self.loc_result = f"D:/RE100_NEXTgroup/result"
        # self.date_result = "230130"
        #
        # self.set_result_loc(opt='부가비용')
        # self.loc_plot = f"D:/RE100_NEXTgroup/plot"
        # self.loc_excel = f"D:/RE100_NEXTgroup/excel"


        # self.contract_voltage = 154


        self.tariff_pre_choice = ['선택1', '선택2', '선택3']


        self.gaprel = 0.001
        self.unit = 10 ** 8
        self.Big_M = 10 ** 8


        self.year_resolution = 1
        self.year_overlap = 0

        """
        발전지역별 송전이용요금단가
        * 발전 배전저압, 배전고압 비용 X (PPA 송배전망 이용요금 산정 계산기 참고함_솔라커넥트)
        1: 수도권 북부지역
        2: 수도권 남부지역
        3: 비수도권 지역
        4: 제주지역
        """
        # self.region_gen = 3
        """
        수요지역별 송전이용요금단가
        1: 수도권지역
        2: 비수도권지역
        3: 제주지역
        """
        # self.region_load = 2

        self.scenario_num = 1
        self.pp_num = 1000




        """
        신규 입력
        """
        self.loc_input_data = f"data/input_parameters_raw.xlsx"
        self.loc_tariff_table = f"data/tariff_table.xlsx"
        self.loc_dema_raw1 = f"data/업종별 에너지 사용 패턴 데이터_200831.xlsx"
        self.loc_dema_raw2 = f"data/AMR고객별부하현황_세부업종별_2019.csv"

        self.storages_dict = {
            'BES': {
                "4h": 4
            }
        }

        self.span_dict = {
            'BES': {
                '4h': 20,
                '8h': 20,
                '12h': 20,
                '20h': 20
            },
            'SG': {
                'pv': 15,
                'wt_onshore': 15
            },
            'PPA': {
                'pv': 15,
                'wt_onshore': 15
            }
        }

        self.cost_item_dict = {
            'PPA': [
                'Use',
                'NT_use',
                'NT_capa',
                'Uplift',
                'TLT'
            ],
            'Tariff': [
                'Use',
                'Capa',
                'CCEC',
                'FCPTAR'
            ]
        }

        self.tariff_set = [
            'pre',
            'pro'
        ]

        self.self_generator_set = [
            'pv',
            'wt_onshore'
        ]

        self.ppa_generator_set = [
            'pv',
            'wt_onshore'
        ]

        self.generator_set = [
            'pv',
            'wt_onshore'
        ]

        self.fund_set = [
            'PPA',
            'TrF'
        ]

        self.tax_set = [
            'PPA',
            'TrF'
        ]

        self.ueac_set = [
            'REC'
        ]

        self.term_contract_PPA = 15

        self.year0 = 2022
        self.year1 = 2040

        self.model_start_y = 2024
        self.model_end_y = 2024
        self.model_start_d = 1
        self.model_end_d = 365
        self.model_start_h = 1
        self.model_end_h = 24
        self.set_y = np.arange(self.year0, self.year1 + 1, 1).astype('int').tolist()
        self.set_d = np.arange(1, 365 + 1, 1).astype('int').tolist()
        self.set_h = np.arange(1, 24 + 1, 1).astype('int').tolist()

        self.set_options()



    # def set_result_loc(self, opt):
    #     opt_name = self.set_pp_name(opt)
    #     self.set_result_dir(opt_name, self.loc_result, self.date_result)
    #     self.loc_pp_result = f"{self.loc_result}/{self.date_result}/{opt_name}"
    #
    # def set_result_dir(self, opt_name, loc_result, date_result):
    #     if os.path.isdir(f"{loc_result}/{date_result}"):
    #         pass
    #     else:
    #         os.makedirs(f"{loc_result}/{date_result}")
    #     if os.path.isdir(f"{loc_result}/{date_result}/{opt_name}"):
    #         pass
    #     else:
    #         os.makedirs(f"{loc_result}/{date_result}/{opt_name}")

    # def set_pp_name(self, opt):
    #     if opt == '부가비용':
    #         if self.customize_exemption_flag:
    #             name = '부가비용면제'
    #         else:
    #             name = '부가비용고려'
    #     return name

    def set_options(self):
        """
        Fix: RE100 달성을 미리 정해진 비중을 반드시 따르면서 달성
        Free: RE100 달성을 미리 정해진 비중을 따르되, 이 이상을 자유롭게 달성 가능
        """
        self.opt_RE100_achievement = 'Fix'

        """
        사회적할인율
        """
        self.rate_discount = 0.045

        """
        불확실성 범위 조절 파라미터
        Range of Uncertainty
        """
        self.RoU_TrF = 100
        self.RoU_LCOE = 100

        """
        망요금 단가 산정을 위한 수용가 및 기업PPA 발전사업의 위치 설정
        Local: 비수도권
        Metro: 수도권
        """
        self.vol_region_load = 154
        self.region_load = 'Local'
        self.region_gen = 'Local'

        """
        손실정산금 산정을 위한 발전 및 수용가의 송배전 사용여부 설정
        """
        self.Use_gen_TN = True
        self.Use_load_DN_high = True

        """
        한전 요금제 선정
        """
        self.tariff_ppa = '기업PPA,고압B' if self.vol_region_load == 154 else '기업PPA,고압C' if self.vol_region_load >= 345 else '기업PPA,고압A'
        self.tariff_pre = '을,고압B' if self.vol_region_load == 154 else '을,고압C' if self.vol_region_load >= 345 else '을,고압A'





class ReadInputData:
    def __init__(self, options: ProbOptions):
        print("reading input data")
        self.generation_sample_g = dict()
        for g in options.generator_set:
            with open(f"data/generation_sample_{g}.json", "r") as f:
                self.generation_sample_g[g] = json.load(f)

        self.cap_max_sg_g = dict()
        self.cap_min_sg_g = dict()
        self.cap_max_ppa_g = dict()
        self.cap_min_ppa_g = dict()
        year_raw_data = pd.read_excel(f"{options.loc_input_data}", sheet_name='year', index_col=0)
        for g in options.generator_set:
            self.cap_max_sg_g[g] = year_raw_data.loc[f"SG_{g}_max", :]
            self.cap_min_sg_g[g] = year_raw_data.loc[f"SG_{g}_min", :]
            self.cap_max_ppa_g[g] = year_raw_data.loc[f"PPA_{g}_max", :]

        self.tariff_table_ppa = pd.read_excel(f"{options.loc_tariff_table}", sheet_name=options.tariff_ppa, index_col=0)

        unit_price_data = pd.read_excel(f"{options.loc_input_data}", sheet_name='unit_price', index_col=0)
        self.tariff_avg = unit_price_data.loc['tariff', '단가']
        self.rec_avg = unit_price_data.loc['rec', '단가']
        self.uplift_avg = unit_price_data.loc['Uplift', '단가']
        self.ccec_avg = unit_price_data.loc['CCEC', '단가']
        self.fcptar_avg = unit_price_data.loc['FCPTAR', '단가']
        self.fund_rate = unit_price_data.loc['Fund', '단가']
        self.tax_rate = unit_price_data.loc['Tax', '단가']

        self.ppa_avg_g = dict()
        self.CPX_avg_g = dict()
        self.OPX_avg_g = dict()
        for g in options.generator_set:
            self.ppa_avg_g[g] = unit_price_data.loc[f'ppa_{g}', '단가']
            self.CPX_avg_g[g] = unit_price_data.loc[f'CPX_{g}', '단가']
            self.OPX_avg_g[g] = unit_price_data.loc[f'OPX_{g}', '단가']

        self.tariff_rate = pd.read_excel(f"{options.loc_input_data}", sheet_name='tariff_rate', index_col=0).loc[:, '단가']
        self.rec_rate = pd.read_excel(f"{options.loc_input_data}", sheet_name='rec_rate', index_col=0).loc[:, 'rate']

        self.lcoe_rate_g = dict()
        for g in options.generator_set:
            self.lcoe_rate_g[g] = pd.read_excel(f"{options.loc_input_data}", sheet_name='lcoe_rate', index_col=0).loc[[f"{g}_min", f"{g}_max"], "rate"]


        # customized
        demand_fac_dict = dict()
        demand_fac_dict['자동차 부품'] = pd.read_excel(f"{options.loc_dema_raw1}", sheet_name='자동차부품', index_col=0, header=0)['Power consumption'].values.reshape(365, 24)
        demand_fac_dict['일반기계 등'] = pd.read_excel(f"{options.loc_dema_raw1}", sheet_name='일반기계', index_col=0, header=0)['Power consumption'].values.reshape(365, 24)
        demand_fac_dict['SK데이터센터'] = pd.read_excel(f"{options.loc_dema_raw1}", sheet_name='데이터센터', index_col=0, header=0)['Power consumption'].values.reshape(365, 24)

        dema_raw2_df = pd.read_csv(f"{options.loc_dema_raw2}", encoding='euc-kr')
        demand_fac_dict['OCI'] = dema_raw2_df[dema_raw2_df['업종'].str.contains('화학물질')].iloc[:, 3:].values
        demand_fac_dict['수소생산클러스터'] = dema_raw2_df[dema_raw2_df['업종'].str.contains('전기, 가스')].iloc[:, 3:].values

        for k in demand_fac_dict.keys():
            demand_fac_dict[k] = demand_fac_dict[k] / np.max(demand_fac_dict[k])

        demand_info = pd.read_excel(f"{options.loc_dema_raw1}",
                                    sheet_name='업종별 수요정보',
                                    index_col=0,
                                    header=0).loc[['SK데이터센터', 'OCI', '자동차 부품', '일반기계 등', '수소생산클러스터'], range(options.year0, options.year1 + 1)]

        self.demand_arr_y_d_h = np.zeros((options.year1 - options.year0 + 1, 365, 24))
        for k in demand_fac_dict.keys():
            self.demand_arr_y_d_h += self.make_pattern(options, demand_fac_dict[k], demand_info.loc[k, :]) * 1000 # kWh


        self.tariff_pre_txt_dict = dict()
        ctp = self.choice_tariff_pre(options, self.demand_arr_y_d_h)
        for y in options.set_y:
            v = ctp[y]
            self.tariff_pre_txt_dict[y] = f"{options.tariff_pre},{v}"

        self.tariff_pre_dict = dict()
        self.tariff_pro_dict = dict()
        self.tariff_pre_dema_dict = dict()
        self.tariff_pro_dema_dict = dict()

        for y in options.set_y:
            tariff_table_pre = pd.read_excel(f"{options.loc_tariff_table}", sheet_name=f"{self.tariff_pre_txt_dict[y]}", index_col=0)
            self.tariff_pre_dict[y] = self.make_tariif_arr(tariff_table_pre)
            self.tariff_pre_dema_dict[y] = float(tariff_table_pre.loc['기본요금', '여름철'])

            tariff_table_pro = pd.read_excel(f"{options.loc_tariff_table}", sheet_name=options.tariff_ppa, index_col=0)
            self.tariff_pro_dict[y] = self.make_tariif_arr(tariff_table=tariff_table_pro)
            self.tariff_pro_dema_dict[y] = float(tariff_table_pro.loc['기본요금', '여름철'])

    def choice_tariff_pre(self, options: ProbOptions, demand_arr_y_d_h):
        choice_dict = dict()
        for y in options.set_y:
            dummy_dict = dict()

            y_tilda = y - options.year0
            for tbc in options.tariff_pre_choice:
                tariff_pre = pd.read_excel(f"{options.loc_tariff_table}", sheet_name=f"{options.tariff_pre},{tbc}", index_col=0)
                p_dema = tariff_pre.loc['기본요금', '여름철']
                tariff_pre_arr = self.make_tariif_arr(tariff_pre)

                dummy_dict[tbc] = np.sum(demand_arr_y_d_h[y_tilda, :, :] * tariff_pre_arr) + \
                                  np.max(demand_arr_y_d_h[y_tilda, :, :]) * p_dema * 12

            choice_dict[y] = [k for k, v in dummy_dict.items() if min(dummy_dict.values()) == v][0]

        return choice_dict

    def make_pattern(self, options, factor, amount_list):
        frame = np.zeros((options.year1 - options.year0 + 1, 365, 24))  # 윤년 제외

        index_1 = 0
        for y in options.set_y:
            frame[index_1, :, :] = factor * (amount_list[y] / np.sum(factor))
            index_1 += 1

        return frame

    def make_tariif_arr(self, tariff_table):
        table = pd.DataFrame(np.zeros((365, 24)), columns=np.arange(1, 25, 1))

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

        return table.values

class SetTime:
    def __init__(self, options: ProbOptions):
        set_year = []
        for y in range(options.model_start_y, options.model_end_y + 1):
            set_year.append(y)
        set_day = []
        for d in range(options.model_start_d, options.model_end_d + 1):
            set_day.append(d)
        set_hour = []
        for h in range(options.model_start_h, options.model_end_h + 1):
            set_hour.append(h)
        self.set_year = set_year
        self.set_day = set_day
        self.set_hour = set_hour

        set_historic_year = []
        for y in range(options.year0, options.model_start_y, 1):
            set_historic_year.append(y)
        self.set_historic_year = set_historic_year


class ParameterPyomoForm:
    def __init__(self, options: ProbOptions, IFN: ReadInputData):
        self.set_storage = list(options.storages_dict.keys())
        self.set_sg_generator = options.self_generator_set
        self.set_ppa_generator = options.ppa_generator_set
        self.set_generator = options.generator_set
        self.set_ppa_item = options.cost_item_dict['PPA']
        self.set_trf_item = options.cost_item_dict['Tariff']
        self.set_tariff = options.tariff_set
        self.set_fund = options.fund_set
        self.set_tax = options.tax_set
        self.set_ueac = options.ueac_set
        self.set_capa = list(options.span_dict.keys())

        self.cal_yearly_disc_rate(options)
        self.cal_rate_lcoe(series=IFN.lcoe_rate_g, RoU=options.RoU_LCOE)
        self.cal_rate_TrF(options,
                          series=IFN.tariff_rate[[2015, 2016, 2017, 2018, 2019, 2020]],
                          init=IFN.tariff_rate[2020],
                          RoU=options.RoU_TrF)

        self.set_tariff_y_d_h(options, IFN)
        self.set_SG_y(options, IFN)
        self.set_PPA_y(options, IFN)
        self.set_nt_y(options)
        self.set_TLT_y(options, IFN)
        self.set_factor_g_y_d_h(options, IFN)
        self.set_achievement_ratio()
        self.set_N_y_d_h(options)

        self.rate_RVR_SG_g_y = self.set_RVR_y(options, source='SG')
        self.rate_RVR_ESS_s_y = self.set_RVR_y(options, source='BES')

        self.price_REC_y = self.set_rec_y(options,
                                          rate_set=IFN.rec_rate,
                                          init=IFN.rec_avg)
        self.price_Uplift_y = self.value_to_dict(options, value=IFN.uplift_avg)
        self.price_CCEC_y = self.value_to_dict(options, value=IFN.ccec_avg)
        self.price_FCPTAR_y = self.value_to_dict(options, value=IFN.fcptar_avg)

        self.demand_y_d_h = self.arr_to_dict_ydh(options, arr=IFN.demand_arr_y_d_h)
        self.capa_max_SG_g_y = self.series_to_dict_g_y(options, series=IFN.cap_max_sg_g)
        self.capa_min_SG_g_y = self.series_to_dict_g_y(options, series=IFN.cap_min_sg_g)
        self.capa_max_PPA_g_y = self.series_to_dict_g_y(options, series=IFN.cap_max_sg_g)
        self.capa_min_PPA_g_y = self.series_to_dict_g_y(options, series=IFN.cap_min_sg_g)

        self.rate_Fund_y = self.value_to_dict(options, value=IFN.fund_rate)
        self.rate_Tax_y = self.value_to_dict(options, value=IFN.tax_rate)



    def set_tariff_y_d_h(self, options: ProbOptions, IFN: ReadInputData):
        self.price_TrF_y = self.cal_trend(options,
                                          trend_type='linear',
                                          init_value=IFN.tariff_avg,
                                          target_rate=self.rate_TrF,
                                          init_rate=1)
        TrF_pre_arr = self.cal_tariff_table(options,
                                            tariff_y=self.price_TrF_y,
                                            tariff_average=IFN.tariff_avg,
                                            tariff_dict=IFN.tariff_pre_dict)
        TrF_pro_arr = self.cal_tariff_table(options,
                                            tariff_y=self.price_TrF_y,
                                            tariff_average=IFN.tariff_avg,
                                            tariff_dict=IFN.tariff_pro_dict)

        self.price_TrF_Use_pre_y_d_h = self.arr_to_dict_ydh(options, arr=TrF_pre_arr)
        self.price_TrF_Use_pro_y_d_h = self.arr_to_dict_ydh(options, arr=TrF_pro_arr)
        self.price_TrF_Capa_pre_y = IFN.tariff_pre_dema_dict
        self.price_TrF_Capa_pro_y = IFN.tariff_pro_dema_dict

    def set_SG_y(self, options: ProbOptions, IFN: ReadInputData):
        self.price_CPX_g_y = dict()
        self.price_OPX_g_y = dict()

        for g in self.set_sg_generator:
            self.price_CPX_g_y[g] = self.cal_trend(options,
                                                   trend_type='linear',
                                                   init_value=IFN.CPX_avg_g[g],
                                                   target_rate=self.rate_lcoe_g[g],
                                                   init_rate=1)
            self.price_OPX_g_y[g] = self.cal_trend(options,
                                                   trend_type='linear',
                                                   init_value=IFN.OPX_avg_g[g],
                                                   target_rate=1,
                                                   init_rate=1)

    def set_PPA_y(self, options: ProbOptions, IFN: ReadInputData):
        self.price_PPA_Use_g_y = dict()
        for g in self.set_ppa_generator:
            self.price_PPA_Use_g_y[g] = self.cal_trend(options=options,
                                              trend_type='linear',
                                              init_value=IFN.ppa_avg_g[g],
                                              target_rate=self.rate_lcoe_g[g],
                                              init_rate=1)

    def set_rec_y(self, options, rate_set, init):
        set_dict = dict()
        rec_min = rate_set.min()
        rec_max = rate_set.max()
        for y in options.set_y:
            if y == options.year0:
                set_dict[y] = float(init)
            else:
                ran_val = np.random.uniform(rec_min, rec_max)
                set_dict[y] = float(set_dict[y - 1] * ran_val)
        return set_dict

    def set_nt_y(self, options: ProbOptions):
        nt_Capa = 0
        nt_Use = 0
        if options.vol_region_load < 0.6:
            nt_Capa += 1066
            nt_Use += 11.33
        elif 0.6 < options.vol_region_load <= 22.9:
            nt_Capa += 548
            nt_Use += 3.05
        else:
            pass

        nt_Capa += 667.61
        if options.region_load == 'Local':
            nt_Use += 1.42
        elif options.region_load == 'Metro':
            nt_Use += 2.44

        nt_Capa += 667.36
        if options.region_gen == 'Local':
            nt_Use += 1.92
        elif options.region_gen == 'Metro':
            nt_Use += 1.20

        self.price_NT_Use_y = self.value_to_dict(options, value=nt_Use)
        self.price_NT_Capa_y = self.value_to_dict(options, value=nt_Capa)

    def set_TLT_y(self, options: ProbOptions, IFN: ReadInputData):
        if options.Use_gen_TN and options.Use_load_DN_high == False:
            loss = 1.57
        elif options.Use_gen_TN and options.Use_load_DN_high:
            loss = 3.54
        elif options.Use_gen_TN == False and options.Use_load_DN_high == False:
            loss = 1.99
        self.price_TLT_y = self.value_to_dict(options, value=loss)

        return loss / 100 * IFN.tariff_avg

    def set_factor_g_y_d_h(self, options: ProbOptions, IFN: ReadInputData):
        self.factor_g_y_d_h = dict()
        for g in self.set_generator:
            self.factor_g_y_d_h[g] = dict()
            sample_set = IFN.generation_sample_g[g]
            for y in options.set_y:
                for d in options.set_d:
                    for h in options.set_h:
                        mean = np.mean(sample_set[f"{d-1},{h-1}"])
                        std = np.std(sample_set[f"{d-1},{h-1}"])
                        while True:
                            val_ = np.random.normal(loc=mean, scale=std)
                            if val_ >= 0:
                                break
                            else:
                                pass
                        self.factor_g_y_d_h[g][y, d, h] = val_
    def set_achievement_ratio(self):
        rate_achievement = dict()
        rate_achievement.update(self.cal_ratio_by_trend('linear', 0, 0, 2022, 2024))
        rate_achievement.update(self.cal_ratio_by_trend('linear', 18.55 / 100, 0, 2024, 2029))
        rate_achievement.update(self.cal_ratio_by_trend('linear', 60 / 100, 18.55 / 100, 2029, 2035))
        rate_achievement.update(self.cal_ratio_by_trend('linear', 100 / 100, 60 / 100, 2035, 2040))
        self.rate_achievement_y = rate_achievement

    def set_RVR_y(self, options: ProbOptions, source):
        """
        :return: 연도별 남아있는 잔존가치 비율
        """
        rate_RVR_k_y = dict()
        for k in options.span_dict[source].keys():
            rate_RVR_k_y[k] = dict()
            for y in options.set_y:
                rate_RVR_k_y[k][y] = max((y + options.span_dict[source][k] - 1) - options.year1, 0) / \
                                          options.span_dict[source][k]
        return rate_RVR_k_y

    def set_N_y_d_h(self, options: ProbOptions):
        self.N_y_d_h = dict()
        if len(options.set_d) == 365 and len(options.set_h) == 24:
            for y in options.set_y:
                for d in options.set_d:
                    for h in options.set_h:
                        self.N_y_d_h[y, d, h] = 1
    def arr_to_dict_ydh(self, options: ProbOptions, arr):
        dict_y_d_h = dict()
        for y in options.set_y:
            y_tilda = y - options.year0
            for d in options.set_d:
                for h in options.set_h:
                    dict_y_d_h[y, d, h] = float(arr[y_tilda, d - 1, h - 1])
        return dict_y_d_h

    def series_to_dict_g_y(self, options: ProbOptions, series):
        dict_g_y = dict()
        for g in series.keys():
            dict_g_y[g] = dict()
            for y in options.set_y:
                dict_g_y[g][y] = float(series[g][y])
        return dict_g_y

    def value_to_dict(self, options, value):
        dict_y = dict()
        for y in options.set_y:
            dict_y[y] = float(value)
        return dict_y

    def cal_yearly_disc_rate(self, options: ProbOptions):
        self.rate_discount_y = dict()
        for y in options.set_y:
            self.rate_discount_y[y] = 1 / (1 + options.rate_discount) ** (y - options.year0)

    def cal_rate_lcoe(self, series, RoU):
        self.rate_lcoe_g = dict()
        for g in self.set_generator:
            rate_lcoe_min_g = series[g][f"{g}_min"]
            rate_lcoe_max_g = series[g][f"{g}_max"]
            self.rate_lcoe_g[g] = self.cal_random_ratio(rate_lcoe_min_g, rate_lcoe_max_g, RoU)

    def cal_rate_TrF(self, options: ProbOptions, series, init, RoU):
        self.rate_TrF_max = self.cal_value_using_regression(options,
                                                            data=series,
                                                            init=init)
        self.rate_TrF_min = 1

        self.rate_TrF = self.cal_random_ratio(self.rate_TrF_min, self.rate_TrF_max, RoU)

    def cal_value_using_regression(self, options, data, init):
        line_fitter = LinearRegression()
        x_data = np.arange(1, data.shape[0] + 1, 1).reshape(-1, 1)
        line_fitter.fit(X=x_data, y=data)

        x_predict = np.arange(x_data[-1][0] + 1, x_data[-1][0] + options.year1 - options.year0 + 1 + 2).reshape(-1, 1)
        y_predict = line_fitter.coef_ * x_predict + line_fitter.intercept_
        R = (y_predict / init).reshape(-1)
        return R[-1]

    def cal_random_ratio(self, r0, r1, per):
        new_r0 = r0 + (r1 - r0) * (1 - per / 100) / 2
        new_r1 = r1 - (r1 - r0) * (1 - per / 100) / 2

        return np.random.uniform(new_r0, new_r1)

    def cal_trend(self, options: ProbOptions, trend_type, init_value, target_rate, init_rate):

        set_dict = dict()
        slope = (target_rate - init_rate) / (options.year1 - options.year0)
        intercept = init_rate - slope * options.year0

        for y in options.set_y:
            if trend_type == 'linear':
                set_dict[y] = float(init_value * (slope * y + intercept))
            elif trend_type == 'constant':
                set_dict[y] = float(init_value)

        return set_dict

    def cal_tariff_table(self, options, tariff_y, tariff_average, tariff_dict):
        axis_new = 0
        for y in options.set_y:
            rate = tariff_y[y] / tariff_average
            if axis_new == 0:
                table = np.expand_dims(tariff_dict[y], axis=0)
            else:
                table = np.append(table, np.expand_dims(tariff_dict[y] * rate, axis=0), axis=0)

            axis_new += 1
        return table


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


class ParameterPulpForm:

    def __init__(self, options: ProbOptions, IFN: ReadInputData):
        self.discount_rate = 0.045
        self.pv_life_span = 20
        self.onshore_life_span = 20

        self.demand_y_d_h = self.arr_to_dict(options, arr=IFN.demand_arr_y_d_h)
        self.ee_y_d_h = self.arr_to_dict(options, arr=IFN.eff_arr_y_d_h)

        self.lambda_ee_y = self.df_to_dict_y(options, IFN.ee_cost, '비용(원)')
        if options.customize_self_generation_flag:
            self.cap_max_sg_pv_y = self.customize_sg_cap(options, IFN.Apv_raw, self.demand_y_d_h, self.ee_y_d_h)
            self.cap_min_sg_pv_y = self.customize_sg_cap(options, IFN.Apv_raw, self.demand_y_d_h, self.ee_y_d_h)
            self.cap_max_sg_onshore_y = self.none_to_dict_y(options, 0)  # 일단, 자가발전으로 육상풍력  미고려
            self.cap_min_sg_onshore_y = self.none_to_dict_y(options, 0)

        else:
            self.cap_max_sg_pv_y = self.df_to_dict_y(options, IFN.cap_max_sg, 'pv')  # 산단 부지 내 설치 가능한 태양광 자가발전 최대용량
            self.cap_min_sg_pv_y = self.none_to_dict_y(options, 0)
            self.cap_max_sg_onshore_y = self.df_to_dict_y(options, IFN.cap_max_sg, 'onshore')  # 산단 부지 내 설치 가능한 육상풍력 자가발전 최대용량
            self.cap_min_sg_onshore_y = self.none_to_dict_y(options, 0)

        self.cap_max_ppa_pv_y = self.df_to_dict_y(options, IFN.cap_max_ppa, 'pv')  # 기업PPA with PV 계약 가능한 용량
        self.cap_max_ppa_onshore_y = self.df_to_dict_y(options, IFN.cap_max_ppa, 'onshore')  # 기업PPA with onshore 계약 가능한 용량

        self.ratio_commission_tariff_ratio_per_won_y = self.fill_value_to_dict(options, 0.1)  # 부가가치세
        self.ratio_tariff_funding_y = self.fill_value_to_dict(options, 0.037)  # 전력산업기반기금 ratio. 모든 항목의 합 * ratio. 출처: 한전

        self.rate_min_pv = IFN.lcoe_rate.loc['pv0', 'rate']
        self.rate_max_pv = IFN.lcoe_rate.loc['pv1', 'rate']
        self.rate_min_onshore = IFN.lcoe_rate.loc['onshore0', 'rate']
        self.rate_max_onshore = IFN.lcoe_rate.loc['onshore1', 'rate']

        self.rate_max_tariff = self.cal_value_using_regression(options,
                                                               data=IFN.tariff_rate.loc[2015:, 'USD/kWh'].values.reshape(-1, 1),
                                                               init=IFN.tariff_rate.loc[2020, 'USD/kWh'])  # year1까지 평균 전기요금 증가율 도출 (과거 데이터 기반(2015년 이상) 선형추세선 적용)
        self.rate_min_tariff = 1

        self.rate_pv = self.cal_random_ratio(self.rate_min_pv, self.rate_max_pv, 100)
        self.rate_onshore = self.cal_random_ratio(self.rate_min_onshore, self.rate_max_onshore, 100)
        self.rate_tariff = self.cal_random_ratio(self.rate_min_tariff, self.rate_max_tariff, 100)

        self.tariff_average = IFN.tariff_avg_2020['won/kWh'].iloc[0]  # 2020년 평균 산업용 전기요금 단가, 환율: 1342.81원/$
        self.tariff_y = self.cal_trend(options, 'linear', self.tariff_average, self.rate_tariff, 1)

        tariff_pre_arr = self.cal_tariff_table(options, self.tariff_y, self.tariff_average, IFN.tariff_pre_dict)
        tariff_pro_arr = self.cal_tariff_table(options, self.tariff_y, self.tariff_average, IFN.tariff_pro_dict)
        self.lambda_tariff_pre_y_d_h = self.arr_to_dict(options, tariff_pre_arr)
        self.lambda_tariff_pro_y_d_h = self.arr_to_dict(options, tariff_pro_arr)
        self.lambda_tariff_dema_pre_y = IFN.tariff_pre_dema_dict
        self.lambda_tariff_dema_pro_y = IFN.tariff_pro_dema_dict

        self.p_pv_install = IFN.inst_cost_avg.loc['pv', 'won/kw']
        self.lambda_CAPEX_PV_y = self.cal_trend(options, 'linear', self.p_pv_install, self.rate_pv, 1)
        self.lambda_OPEX_PV_y = self.cal_trend(options, 'linear', self.p_pv_install * 0.02, self.rate_pv, 1)  # CAPEX의 2% 가정

        self.p_onshore_install = IFN.inst_cost_avg.loc['onshore', 'won/kw']
        self.lambda_CAPEX_onshore_y = self.cal_trend(options, 'linear', self.p_onshore_install, self.rate_onshore, 1)
        self.lambda_OPEX_onshore_y = self.cal_trend(options, 'linear', self.p_onshore_install * 0.02, self.rate_onshore, 1)  # CAPEX의 2% 가정

        self.lambda_PPA_pv_y = self.cal_trend(options=options,
                                              trend_type='linear',
                                              init_value=IFN.ppa_avg.loc['pv', 'won'],
                                              target_rate=self.rate_pv,
                                              init_rate=1)

        self.lambda_PPA_onshore_y = self.cal_trend(options=options,
                                                   trend_type='linear',
                                                   init_value=IFN.ppa_avg.loc['onshore', 'won'],
                                                   target_rate=self.rate_onshore,
                                                   init_rate=1)

        self.lambda_eac_y = self.make_rec_set(options, IFN.rec_rate['rate'].values,
                                              IFN.rec_avg['won/kWh'].iloc[0])

        # 22.10.11 신규 항목 추가 (복지및특례비용, 기후환경요금, 연료비조정액)
        self.lambda_welfare_y = self.fill_value_to_dict(options, 0)  # 22.10.11 한전 엔터 에너지마켓플레이스 참고, 22.12.01 복지 및 특례할인은 제3자 PPA만 부과
        self.lambda_climate_y = self.fill_value_to_dict(options, 9)  # 공급약관 및 세칙 개정 사항 안내('22.12.30) 참고
        self.lambda_fuel_adjustment_y = self.fill_value_to_dict(options, 5)  # 22.10.11 한전 전기요금계산기 참고

        self.Apv_d_h = self.np_to_dict_d_h(options, IFN.Apv_raw)
        self.Aonshore_d_h = self.np_to_dict_d_h(options, IFN.Aonshore_raw)

        lambda_nt_d, lambda_nt_c = self.cal_network_tariff(options)  # nt_d: won/kW, nt_c: won/kWh. 출처: 한전송배전망이용요금표
        lambda_AS_payment = float(IFN.as_payment_avg.loc['AS payment', 'won/kWh'])

        lambda_loss_payment = self.cal_loss_payment(options, self.tariff_average)  # 손실정산금 단가 won/kWh. 손실율 * 평균전기요금단가(2020 산업용 전기요금 단가)
        self.ratio_ppa_funding_y = self.fill_value_to_dict(options, 0.037)  # 전력산업기반기금 ratio. 모든 항목의 합 * ratio. 출처: 한전
        self.ratio_commission_ppa_ratio_per_won_y = self.fill_value_to_dict(options, 0.1)

        self.lambda_loss_payment_y = self.fill_value_to_dict(options, lambda_loss_payment)
        self.lambda_AS_payment_y = self.fill_value_to_dict(options, lambda_AS_payment)
        self.lambda_nt_c_y = self.fill_value_to_dict(options, lambda_nt_c)

        self.print_status()

    def fill_value_to_dict(self, options, value):
        set_dict = dict()
        for y in options.set_y:
            set_dict[y] = float(value)
        return set_dict

    def customize_sg_cap(self, options, A_file, demand_y_d_h, ee_y_d_h):
        cap_sg_y = dict()

        for y in options.set_y:
            demand = 0
            ee = 0
            for d in options.set_d:
                for h in options.set_h:
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

    def arr_to_dict(self, options, arr):
        dict_y_d_h = dict()
        for y in options.set_y:
            y_tilda = y - options.year0
            for d in options.set_d:
                for h in options.set_h:
                    dict_y_d_h[y, d, h] = float(arr[y_tilda, d - 1, h - 1])

        return dict_y_d_h

    def none_to_dict_y_d_h(self, options, value):
        set_dict = dict()
        for y in options.set_y:
            for d in options.set_d:
                for h in options.set_h:
                    set_dict[y, d, h] = float(value)
        return set_dict

    def none_to_dict_y(self, options, value):
        set_dict = dict()
        for y in options.set_y:
            set_dict[y] = float(value)
        return set_dict

    def df_to_dict_y(self, options, file, item):
        set_dict = dict()
        for y in options.set_y:
            set_dict[y] = float(file.loc[item, y])
        return set_dict

    def np_to_dict_d_h(self, options, np_file):
        set_dict = dict()
        for d in options.set_d:
            for h in options.set_h:
                set_dict[d, h] = float(np_file[d - 1, h - 1])

        return set_dict

    def cal_value_using_regression(self, options, data, init):
        line_fitter = LinearRegression()
        x_data = np.arange(1, data.shape[0] + 1, 1).reshape(-1, 1)
        line_fitter.fit(X=x_data, y=data)

        x_predict = np.arange(x_data[-1][0] + 1, x_data[-1][0] + options.year1 - options.year0 + 1 + 1).reshape(-1, 1)
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

        set_dict = dict()
        slope = (target_rate - init_rate) / (options.year1 - options.year0)
        intercept = init_rate - slope * options.year0

        for y in options.set_y:
            if trend_type == 'linear':
                set_dict[y] = float(init_value * (slope * y + intercept))
            elif trend_type == 'constant':
                set_dict[y] = float(init_value)

        return set_dict

    # 공휴일 또는 주말 요금 반영 안함
    def cal_tariff_table(self, options, tariff_y, tariff_average, tariff_dict):
        axis_new = 0
        for y in options.set_y:
            rate = tariff_y[y] / tariff_average
            if axis_new == 0:
                table = np.expand_dims(tariff_dict[y], axis=0)
            else:
                table = np.append(table, np.expand_dims(tariff_dict[y] * rate, axis=0), axis=0)

            axis_new += 1
        return table


    def make_rec_set(self, options, rate_set, init):
        set_dict = dict()
        rec_min = rate_set.min()
        rec_max = rate_set.max()
        for y in options.set_y:
            if y == options.year0:
                set_dict[y] = float(init)
            else:
                ran_val = np.random.uniform(rec_min, rec_max)
                set_dict[y] = float(set_dict[y - 1] * ran_val)
        return set_dict

    def print_status(self):
        print("Setting input parameters is completed")

