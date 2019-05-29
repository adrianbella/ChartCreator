from chart import Chart


def generate_basic_training_unique_charts(ch):
    ch.paint_and_save_reward_chart('VirtualDrone-v0_SARSAAgent')
    ch.paint_and_save_loss_chart('VirtualDrone-v0_SARSAAgent')

    #ch.paint('VirtualDrone-v0_DDQNAgent', basic_options[3])
    #ch.paint('VirtualDrone-v0_DDQNAgent', basic_options[4])
    #ch.paint('VirtualDrone-v0_DDQNAgent', basic_options[0])


    #ch.paint('VirtualDrone-v0_SARSAAgent_conv18_conv216_fc8', master_student_options[0])
    #ch.paint('VirtualDrone-v0_SARSAAgent_conv18_conv216_fc8', master_student_options[2])

    #ch.paint('VirtualDrone-v0_DDQNAgent_conv18_conv216_fc8', master_student_options[0])
    #ch.paint('VirtualDrone-v0_DDQNAgent_conv18_conv216_fc8', master_student_options[2])

    #ch.paint('VirtualDrone-v0_DQNAgent_conv18_conv216_fc8', master_student_options[0])
    #ch.paint('VirtualDrone-v0_DQNAgent_conv18_conv216_fc8', master_student_options[2])

def generate_basic_training_comparation_graphs(ch, plot_style, axes_font_size, line_width):
    source = ['VirtualDrone-v0_DQNAgent', 'VirtualDrone-v0_DDQNAgent', 'VirtualDrone-v0_SARSAAgent']
    rolling = 70000


    ch.piant_three_plots(source, 'reward', rolling, line_width, 'Reward [point/step]', axes_font_size, plot_style)
    ch.piant_three_plots(source, 'losses', rolling, line_width, 'Loss', axes_font_size, plot_style)
    ch.piant_three_plots(source, 'q_values', rolling, line_width, 'Q-values', axes_font_size, plot_style)


def generate_basic_training_epsilone_graph(ch, plot_style, axes_font_size, line_width):
    source = 'epsilone_value_changing'
    columns = ['iteration', 'percent']


    ch.paint_epsilone_plot(source, columns, plot_style, line_width, axes_font_size)


def generate_basic_training_result_chart(ch, plot_style, axes_font_size, line_width, chart_type, yaxis_label):
    columns = ['iteration', 'dqn_' + chart_type, 'ddqn_' + chart_type, 'sarsa_' + chart_type]
    ch.paint_basic_training_result_plot(chart_type, columns, line_width, yaxis_label, axes_font_size, plot_style)


def generate_basic_training_reward_function(ch, plot_style, axes_font_size, line_width):
    source = 'reward_function'
    columns = ['distance', 'reward']

    ch.paint_reward_plot(source, columns, plot_style, line_width, axes_font_size)

def generate_master_student_training_charts(ch, plot_style, axes_font_size, line_width):
    source = ['VirtualDrone-v0_DQNAgent_conv18_conv216_fc8'
        , 'VirtualDrone-v0_DDQNAgent_conv18_conv216_fc8'
        , 'VirtualDrone-v0_SARSAAgent_conv18_conv216_fc8']
    rolling = 1000

    ch.piant_three_plots(source, 'acc', rolling, line_width, 'Accuracy', axes_font_size, plot_style)
    ch.piant_three_plots(source, 'loss', rolling, line_width, 'Loss', axes_font_size, plot_style)

if __name__ == "__main__":
    ch = Chart()

    basic_options = ['reward', 'actions', 'episodes', 'losses'
               , 'q_values', 'epsilone', 'acc', 'loss']
    master_student_options = ['acc', 'batch', 'loss', 'size']

    plot_style = 'classic'
    axes_font_size = 20
    line_width = 3

    #generate_basic_training_unique_charts(ch)
    #generate_basic_training_comparation_graphs(ch, plot_style, axes_font_size, line_width)
    #generate_basic_training_epsilone_graph(ch, plot_style, axes_font_size, line_width)

    chart_types = ['training_accuracy', 'training_episodic_avg_reward', 'training_episodic_avg_step'
                    , 'validation_accuracy', 'validation_episodic_avg_reward', 'validation_episodic_avg_step'
                    , 'test_accuracy', 'test_episodic_avg_reward', 'test_episodic_avg_step']

    yaxis_labels = ['Accuracy', 'Episodic Average Reward', 'Episodic Average Count of Steps']

    for i in range(0, len(chart_types)):
        generate_basic_training_result_chart(ch, plot_style, axes_font_size, line_width, chart_types[i], yaxis_labels[int(i%3)])

    #generate_basic_training_reward_function(ch, plot_style, axes_font_size, line_width)

    #generate_master_student_training_charts(ch, plot_style, axes_font_size, line_width)



