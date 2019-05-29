import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

class Chart:
    def paint_and_save_loss_chart(self, source):
        csv = pd.read_csv(source + '.csv')
        df = pd.DataFrame(csv, columns=['loss'])

        df.plot()

        plt.savefig('./charts/losscharts/' + source)
        plt.close()

    def paint_and_save_reward_chart(self, source):
        csv = pd.read_csv(source + '.csv')
        df = pd.DataFrame(csv, columns=['rewards'])

        df.plot()

        plt.savefig('./charts/rewardcharts/' + source)
        plt.close()

    def paint(self, source, column):
        csv = pd.read_csv(source + '.csv')
        df = pd.DataFrame(csv, columns=[column])
        avg = df.rolling(100).mean()

        plt.figure(figsize=(15, 10))
        plt.plot(avg)
        plt.legend(loc=2)
        plt.savefig('./' + source + '_' + column)
        plt.close()
        plt.show()

    def piant_three_plots(self, source, column, rolling, linewidth, yaxislabel, axes_font_size, plot_style):
        #print(plt.style.available)

        redline_x = np.arange(0, 300000, 1000)
        redline_y = np.arange(0.95, 0.951, 0.000003344)

        dqn_csv = pd.read_csv(source[0] + '.csv')
        dqn_df = pd.DataFrame(dqn_csv, columns=[column])
        dqn_avg = dqn_df.rolling(rolling).mean()

        ddqn_csv = pd.read_csv(source[1] + '.csv')
        ddqn_df = pd.DataFrame(ddqn_csv, columns=[column])
        ddqn_avg = ddqn_df.rolling(rolling).mean()

        sarsa_csv = pd.read_csv(source[2] + '.csv')
        sarsa_df = pd.DataFrame(sarsa_csv, columns=[column])
        sarsa_avg = sarsa_df.rolling(rolling*2).mean()

        # style
        plt.style.use(plot_style)
        # create a color palette
        palette = plt.get_cmap('Set1')

        plt.figure(figsize=(15, 10))
        #plt.xlim(0, 1250000)
        if yaxislabel == 'Accuracy':
            plt.ylim(0,1.1)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.plot(redline_x, redline_y, 'r--', label="95% accuracy")
            plt.legend(loc=4)

        plt.plot(dqn_avg, label="DQN", color=palette(3), linewidth=linewidth)
        plt.plot(ddqn_avg, label="DDQN", color=palette(4), linewidth=linewidth)
        plt.plot(sarsa_avg, label="SARSA", color=palette(1), linewidth=linewidth)

        plt.legend(loc=3)
        plt.xlabel("Iterations", fontsize=axes_font_size)
        plt.ylabel(yaxislabel, fontsize=axes_font_size)
        plt.savefig('./DQN_DDQN_SARSA_master_student_' + column)
        plt.close()
        plt.show()

    def paint_epsilone_plot(self, source, columns, plot_style, linewidth, axes_font_size):
        csv = pd.read_csv(source + '.csv')
        df1 = pd.DataFrame(csv, columns=[columns[0]])
        df2 = pd.DataFrame(csv, columns=[columns[1]])

        # style
        plt.style.use(plot_style)
        # create a color palette
        palette = plt.get_cmap('Set1')
        plt.figure(figsize=(15, 10))
        plt.xlim(0, 1250000)
        plt.ylim(0, 100)
        plt.xticks(np.arange(0, 1250000, 100000))
        plt.yticks(np.arange(0, 110, 10))

        plt.plot(df1, df2, label="Epsilone", color=palette(0), linewidth=linewidth)
        plt.legend(loc=2)
        plt.xlabel("Iterations", fontsize=axes_font_size)
        plt.ylabel("Epsilone value [%]", fontsize=axes_font_size)
        plt.savefig('./Epsilone_value_changing')
        plt.close()
        plt.show()

    def paint_basic_training_result_plot(self, chart_type, columns, line_width, yaxis_label, axes_font_size, plot_style):
        csv = pd.read_csv(chart_type + '.csv')
        iteration_df = pd.DataFrame(csv, columns=[columns[0]])
        dqn_df = pd.DataFrame(csv, columns=[columns[1]])
        ddqn_df = pd.DataFrame(csv, columns=[columns[2]])
        sarsa_df = pd.DataFrame(csv, columns=[columns[3]])

        # style
        plt.style.use(plot_style)
        # create a color palette
        palette = plt.get_cmap('Set1')


        plt.figure(figsize=(15, 10))
        #plt.xticks(np.arange(0, 2300000, 200000))

        if yaxis_label == 'Accuracy':
            plt.ylim(0, 1.1)
            plt.yticks(np.arange(0, 1.1, 0.1))

        elif yaxis_label == 'Episodic Average Reward':
            plt.ylim(-160, 50)
            plt.yticks(np.arange(-160, 50, 10))
        elif yaxis_label == 'Episodic Average Count of Steps':
            plt.ylim(0, 45)
            plt.yticks(np.arange(0, 45, 5))

        plt.plot(iteration_df, dqn_df, label="DQN", color=palette(3), linewidth=line_width)
        plt.plot(iteration_df, ddqn_df, label="DDQN", color=palette(4), linewidth=line_width)
        plt.plot(iteration_df, sarsa_df, label="SARSA", color=palette(1), linewidth=line_width)
        plt.legend(loc=4)

        if chart_type == 'training_accuracy':
            plt.plot([174080], [0.99], marker='o', markersize=20, color='red')
            plt.plot([266240], [0.99], marker='o', markersize=20, color='red')
            plt.plot([112640], [0.98], marker='o', markersize=10, color=palette(1))
        if chart_type == 'validation_accuracy':
            plt.plot([30720], [0.66], marker='o', markersize=10, color=palette(3))
            plt.plot([20480], [0.68], marker='o', markersize=20, color='red')
            plt.plot([40960], [0.64], marker='o', markersize=10, color=palette(1))
        if chart_type == 'test_accuracy':
            plt.plot([20480], [0.71], marker='o', markersize=20, color='red')
            plt.plot([40960], [0.68], marker='o', markersize=10, color=palette(4))
            plt.plot([92160], [0.68], marker='o', markersize=10, color=palette(1))

        if chart_type == 'training_episodic_avg_reward':
            plt.plot([174080], [36.03], marker='o', markersize=20, color='red')
            plt.plot([266240], [35.72], marker='o', markersize=10, color=palette(4))
            plt.plot([112640], [34.38], marker='o', markersize=10, color=palette(1))
        if chart_type == 'validation_episodic_avg_reward':
            plt.plot([30720], [-19.97], marker='o', markersize=10, color=palette(3))
            plt.plot([20480], [-10.2], marker='o', markersize=20, color='red')
            plt.plot([51200], [-22.68], marker='o', markersize=10, color=palette(1))
        if chart_type == 'test_episodic_avg_reward':
            plt.plot([20480], [-0.63], marker='o', markersize=20, color='red')
            plt.plot([40960], [-5.71], marker='o', markersize=10, color=palette(4))
            plt.plot([40960], [-3.29], marker='o', markersize=10, color=palette(1))

        if chart_type == 'training_episodic_avg_step':
            plt.plot([163840], [17.93], marker='o', markersize=20, color='red')
            plt.plot([266240], [18.73], marker='o', markersize=10, color=palette(4))
            plt.plot([112640], [18.15], marker='o', markersize=10, color=palette(1))
        if chart_type == 'validation_episodic_avg_step':
            plt.plot([81920], [37.89], marker='o', markersize=10, color=palette(3))
            plt.plot([51200], [37.21], marker='o', markersize=10, color=palette(1))
            plt.plot([51200], [33.79], marker='o', markersize=20, color='red')
        if chart_type == 'test_episodic_avg_step':
            plt.plot([30720], [34.22], marker='o', markersize=10, color=palette(3))
            plt.plot([40960], [33.76], marker='o', markersize=20, color='red')
            plt.plot([163840], [37.33], marker='o', markersize=10, color=palette(1))

        plt.xlabel("Iterations", fontsize=axes_font_size)
        plt.ylabel(yaxis_label, fontsize=axes_font_size)
        plt.savefig('./DQN_DDQN_SARSA_mast_stud_' + chart_type)
        plt.close()
        plt.show()

    def paint_reward_plot(self, source, columns, plot_style, linewidth, axes_font_size):
        csv = pd.read_csv(source + '.csv')
        df1 = pd.DataFrame(csv, columns=[columns[0]])
        df2 = pd.DataFrame(csv, columns=[columns[1]])

        # style
        plt.style.use(plot_style)
        # create a color palette
        palette = plt.get_cmap('Set1')
        plt.figure(figsize=(15, 10))
        plt.xlim(0, 36)
        plt.ylim(0, 14)
        plt.xticks(np.arange(0, 36, 1))
        plt.yticks(np.arange(0, 14, 1))

        plt.plot(df1, df2, label="Received immediate reward ", color=palette(4), linewidth=linewidth)
        plt.legend(loc=1)
        plt.xlabel("Summed Distance", fontsize=axes_font_size)
        plt.ylabel("Reward", fontsize=axes_font_size)
        plt.savefig('./reward_function')
        plt.close()
        plt.show()
