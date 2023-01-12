import matplotlib.pyplot as plt
import platform
import numpy as np
import math
import copy

from lib.parameters import ParameterPulpFrom, ReadInputData, ProbOptions


class PlotOpt:

    def set_frame(self):
        if platform.system() == 'Windows':
            plt.rc('font', family='Malgun Gothic')
        plt.rcParams['axes.unicode_minus'] = False

        self.fig, self.axes = plt.subplots(figsize=(30, 30))

    def set_axis(self, xlim_min, xlim_max, yticks, ytickslabel, x_label, y_label):
        self.axes.set_xlim([xlim_min - 1, xlim_max + 1])
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(ytickslabel)
        self.axes.set_xlabel(x_label, fontsize=30)
        self.axes.set_ylabel(y_label, fontsize=30)
        self.axes.tick_params(axis='x', labelsize=25)
        self.axes.tick_params(axis='y', labelsize=25)

    def set_axis2(self, xlim_min, xlim_max, xticks, xtickslabel, yticks, ytickslabel, x_label, y_label, ylim_min, ylim_max):
        self.axes.set_xlim([xlim_min - 1, xlim_max + 1])
        self.axes.set_ylim([ylim_min - 1, ylim_max + 1])
        self.axes.set_xticks(xticks)
        self.axes.set_xticklabels(xtickslabel, rotation=45)
        self.axes.set_yticks(yticks)
        self.axes.set_yticklabels(ytickslabel)
        self.axes.set_xlabel(x_label, fontsize=20)
        self.axes.set_ylabel(y_label, fontsize=30)
        self.axes.tick_params(axis='x', labelsize=25)
        self.axes.tick_params(axis='y', labelsize=30)
        self.axes.grid(True, axis='y', alpha=0.5, linestyle='--')

    def set_axis3(self, xlim_min, xlim_max, x_label, y_label):
        self.axes.set_xlim([xlim_min - 1, xlim_max + 1])
        self.axes.set_xlabel(x_label, fontsize=30)
        self.axes.set_ylabel(y_label, fontsize=30)
        self.axes.tick_params(axis='x', labelsize=20)
        self.axes.tick_params(axis='y', labelsize=20)

    def save_fig(self, loc, name):
        self.fig.savefig(f"{loc}/{name}.png")

    def set_title(self, title):
        self.fig.suptitle(title, fontsize=30)

    # def opt1_scatter(self, x_data, y_data):
    #     self.axes.scatter(x_data, y_data)

    def opt1_text(self, x_loc, y_loc, text):
        self.axes.text(x_loc, y_loc, text, fontsize=20)

    def opt2_label_color(self, num):
        if num == 0:
            label = 'Self-generation (PV)'
            color = 'orangered'
        elif num == 1:
            label = 'Self-generation (Onshore Wind)'
            color = 'lightsalmon'
        elif num == 2:
            label = 'Corporate PPAs (PV)'
            color = 'goldenrod'
        elif num == 3:
            label = 'Corporate PPAs (Onshore Wind)'
            color = 'deepskyblue'
        elif num == 4:
            label = 'Unbundled EAC'
            color = 'b'
        elif num == 5:
            label = 'Energy Efficiency'
            color = 'dimgray'

        return label, color

    def opt3_label_color(self, num):
        if num == 0:
            label = 'Case 1'
            color = 'c'
        elif num == 1:
            label = 'Case 2'
            color = 'b'

        return label, color

    def opt3_bar(self, x_data, y_data):
        bottom = 0
        for num in range(2):
            label, color = self.opt3_label_color(num=num)
            if num == 0:
                self.axes.bar(x_data, y_data[:, num], width=0.3, color=color, label=label)
            else:
                bottom += y_data[:, num - 1]
                self.axes.bar(x_data, y_data[:, num], width=0.3, bottom=bottom, color=color, label=label)
        self.fig.legend(loc='upper center', ncol=2, fontsize=15, frameon=True, shadow=True)

    def opt4_bar(self, alpha, color1, label1, color2, label2, country, result_dict, control1, control2, options, fig_name):
        npv_list = []
        BAU_npv_list = []
        for k in result_dict[country].keys():
            npv_list.append(result_dict[country][k]['npv'])
            BAU_npv_list.append(result_dict[country][k]['BAU'])
        npv_array = np.array(npv_list)
        npv_min = np.min(npv_array)
        npv_max = np.max(npv_array)
        BAU_avg = np.sum(BAU_npv_list) / len(BAU_npv_list)

        x_axis_min = math.floor(npv_min / 10 ** (int(math.log10(npv_min)))) * 10 ** (int(math.log10(npv_min)))
        x_axis_max = math.ceil(npv_max / 10 ** (int(math.log10(npv_max)))) * 10 ** (int(math.log10(npv_max)))
        
        if BAU_avg > x_axis_max:
            x_axis_max = BAU_avg + (x_axis_max - x_axis_min) / control1
        else:
            pass
        interval = (x_axis_max - x_axis_min) / control1
        
        npv_x_axis = []
        npv_y_axis = []
        init = copy.deepcopy(x_axis_min)
        for i in range(control1):
            npv_y_axis.append(np.where((npv_array >= init) * (npv_array < init + interval))[0].shape[0])
            npv_x_axis.append(init)
            init += interval
        npv_x_axis.append(init)

        # y축 데이터 -> 0인 데이터는 제외
        for i in range(len(npv_y_axis)):
            v = npv_y_axis[i]
            if v == 0:
                pass
            else:
                print(f"0부터 {i - 1}지점까지 0 데이터")
                break

        for i in reversed(range(len(npv_y_axis))):
            v = npv_y_axis[i]
            if v == 0:
                pass
            else:
                print(f"{len(npv_y_axis)}부터 {i + 1}지점까지 0 데이터")
                break

        choice1 = int(input())
        choice2 = int(input())

        y_new = npv_y_axis[choice1:choice2]
        x_new = npv_x_axis[choice1:choice2+1]

        array_x = np.array(x_new).astype('int').astype('str')
        for i in range(len(x_new)):
            if i % control2 == 0:
                array_x[i] = f"${array_x[i]}"
            else:
                array_x[i] = ''
        array_y = np.arange(0, max(y_new) + 5, 5).astype('str')
        for i in range(array_y.shape[0]):
            if i % 5 == 0:
                pass
            else:
                array_y[i] = ''
                
        self.set_frame()
        self.set_axis2(xlim_min=min(x_new), xlim_max=max(x_new),
                    xticks=x_new,
                    xtickslabel=array_x,
                    yticks=np.arange(0, max(y_new) + 5, 5),
                    ytickslabel=array_y,
                    x_label='', y_label='', ylim_min=0, ylim_max=max(y_new) + 5)

        x_new.pop(-1)
        self.axes.bar(np.array(x_new)+(interval/2), y_new, width=interval/4, color=color1, alpha=alpha, label=label1)
        self.axes.axvline(BAU_avg, color=color2, label=label2)
        self.axes.legend(loc='upper left', ncol=1, fontsize=20, frameon=True, shadow=True)
        self.save_fig(options.loc4, country, fig_name)
    
    def opt5_bar(self, alpha, color1, label1, color2, label2, country, item, size, result_dict, control1, control2, 
                 options, fig_name):
        npv_list = []
        for k in result_dict[f"{country},{item},{size}"].keys():
            npv_list.append(result_dict[f"{country},{item},{size}"][k]['npv'])
        npv_array = np.array(npv_list)
        npv_min = np.min(npv_array)
        npv_max = np.max(npv_array)

        x_axis_min = math.floor(npv_min / 10 ** (int(math.log10(npv_min)))) * 10 ** (int(math.log10(npv_min)))
        x_axis_max = math.ceil(npv_max / 10 ** (int(math.log10(npv_max)))) * 10 ** (int(math.log10(npv_max)))

        interval = (x_axis_max - x_axis_min) / control1

        npv_x_axis = []
        npv_y_axis = []
        init = copy.deepcopy(x_axis_min)
        for i in range(control1):
            npv_y_axis.append(np.where((npv_array >= init) * (npv_array < init + interval))[0].shape[0])
            npv_x_axis.append(init)
            init += interval
        npv_x_axis.append(init)

        # y축 데이터 -> 0인 데이터는 제외
        for i in range(len(npv_y_axis)):
            v = npv_y_axis[i]
            if v == 0:
                pass
            else:
                print(f"0부터 {i - 1}지점까지 0 데이터")
                break

        for i in reversed(range(len(npv_y_axis))):
            v = npv_y_axis[i]
            if v == 0:
                pass
            else:
                print(f"{len(npv_y_axis)}부터 {i + 1}지점까지 0 데이터")
                break
                
        choice1 = int(input())
        choice2 = int(input())
        y_new = npv_y_axis[choice1:choice2]
        x_new = npv_x_axis[choice1:choice2 + 1]

        array_x = np.array(x_new).astype('int').astype('str')
        for i in range(len(x_new)):
            if i % control2 == 0:
                array_x[i] = f"${array_x[i]}"
            else:
                array_x[i] = ''
        array_y = np.arange(0, max(y_new) + 5, 5).astype('str')
        for i in range(array_y.shape[0]):
            if i % 5 == 0:
                pass
            else:
                array_y[i] = ''

        self.set_frame()
        self.set_axis2(xlim_min=min(x_new), xlim_max=max(x_new),
                    xticks=x_new,
                    xtickslabel=array_x,
                    yticks=np.arange(0, max(y_new) + 5, 5),
                    ytickslabel=array_y,
                    x_label='', y_label='', ylim_min=0, ylim_max=max(y_new) + 5)

        x_new.pop(-1)
        self.axes.bar(np.array(x_new)+(interval/2), y_new, width=interval/4, color=color1, alpha=alpha, label=label1)
        self.axes.legend(loc='upper left', ncol=1, fontsize=20, frameon=True, shadow=True)
        self.save_fig(options.loc4, country, fig_name)

    def opt6_bar(self, country, options, color, label, fig_name, result_dict):
        u_list = []
        for k in result_dict[country].keys():
            u_list.append(result_dict[country][k]['u_y'])

        u_set = np.zeros((options.year1 - options.year0 + 1))
        for u in u_list:
            if np.where(u == 1)[0].shape[0] == 0:
                pass
            else:
                u_set[np.where(u == 1)[0][0]] += 1
        self.set_frame()
        self.set_axis(xlim_min=options.year0, xlim_max=options.year1, yticks=np.arange(0, u_set.max() + 100, 100), 
                      ytickslabel=np.arange(0, u_set.max() + 100, 100).astype('str'), x_label='Year', y_label='Frequency')

        x_axis = np.arange(options.year0, options.year1 + 1)
        y_axis = u_set
        self.axes.bar(x_axis, y_axis, color=color, label=label)
        self.axes.legend(loc='upper center', ncol=2, fontsize=15, frameon=True, shadow=True)
        self.save_fig(options.loc4, country, fig_name)


    # def opt5_line(self, x_axis, y_axis, color, label, alpha):
    #     for i in range(len(y_axis)):
    #         if i == 0:
    #             self.axes.plot(x_axis, y_axis[i], color=color, label=label, alpha=alpha)
    #         else:
    #             self.axes.plot(x_axis, y_axis[i], color=color, alpha=alpha)
    # 
    # def opt1_1_line(self, x_axis, y_axis, color, label):
    #     self.axes.plot(x_axis, y_axis, color=color, label=label)
    #     self.axes.legend(loc='upper left', ncol=2, fontsize=15, frameon=True, shadow=True)
    #     self.axes.grid(True, axis='y', alpha=0.5, linestyle='--')
    #     self.axes.axhline(50, color='k')
    # 
    # def opt1_2_bar(self, x_axis, y_axis, color, label):
    #     self.axes.bar(x_axis, y_axis, color=color, label=label)
    #     self.axes.legend(loc='upper center', ncol=2, fontsize=15, frameon=True, shadow=True)
