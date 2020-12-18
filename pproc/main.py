from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from cfdsolver.utils import create_dir

def ax_prop(ax, x, y, label, title):
    ax.plot(x, y, label=label)
    ax.grid(True)
    ax.set_title(title)
    ax.set_yscale('log')

if __name__ == '__main__':    
    fig_dir = 'figures/UNet/nparams/'
    create_dir(fig_dir)

    # data_dir = "../outputs/log/scales/"
    # train_names = {"MSNet3":"/random_8/1202_142147/events.out.tfevents.1606915313.krakengpu1.cluster.173444.0",
    #             "MSNet4":"/random_8/1202_142232/events.out.tfevents.1606915359.krakengpu1.cluster.173589.0",
    #             "MSNet5":"/random_8/1202_142325/events.out.tfevents.1606915410.krakengpu2.cluster.111521.0", 
    #             "MSNet6":"/random_8/1202_142358/events.out.tfevents.1606915445.krakengpu2.cluster.111638.0"}

    # data_dir = "../outputs/log/nparams/"
    # train_names = {"MSNet5_small":"/random_8/1202_183214/events.out.tfevents.1606930341.krakengpu2.cluster.149463.0",
    #             "MSNet5_big":"/random_8/1202_183127/events.out.tfevents.1606930293.krakengpu2.cluster.149333.0",
    #             "MSNet5_big_1":"/random_8/1203_013107/events.out.tfevents.1606955475.krakengpu2.cluster.198491.0",
    #             "MSNet5_big_2":"/random_8/1203_013024/events.out.tfevents.1606955431.krakengpu2.cluster.198397.0"}

    # data_dir = "../outputs/log/nparams/MSNet5_big/"
    # train_names = {"random_4":"/1203_095415/events.out.tfevents.1606985662.krakengpu2.cluster.253121.0",
    #             "random_8":"/1202_183127/events.out.tfevents.1606930293.krakengpu2.cluster.149333.0",
    #             "random_16":"/1203_095526/events.out.tfevents.1606985734.krakengpu2.cluster.253310.0",
    #             "target_case":"/1203_095708/events.out.tfevents.1606985836.krakengpu1.cluster.276814.0"}
    
    # data_dir = "../outputs/log/nparams/MSNet5_big/"
    # train_names = {"fourier_5":"/1203_095630/events.out.tfevents.1606985800.krakengpu1.cluster.276690.0",
    #             "fourier_5_2":"/1203_150636/events.out.tfevents.1607004403.krakengpu1.cluster.319257.0",
    #             "fourier_5_4":"/1203_150833/events.out.tfevents.1607004521.krakengpu1.cluster.319624.0",
    #             "target_case":"/1203_095708/events.out.tfevents.1606985836.krakengpu1.cluster.276814.0"}

    data_dir = "../outputs/log/scales/"
    train_names = {"UNet3":"/random_8/1214_004843/events.out.tfevents.1607903328.krakengpu1.cluster.14719.0",
                "UNet4":"/random_8//1214_004911/events.out.tfevents.1607903358.krakengpu1.cluster.14829.0",
                "UNet5":"/random_8/1214_004944/events.out.tfevents.1607903389.krakengpu2.cluster.312923.0",
                "UNet6":"/random_8/1214_005006/events.out.tfevents.1607903413.krakengpu2.cluster.313030.0",
                "../nparams/MSNet5_big":"/random_8/1202_183127/events.out.tfevents.1606930293.krakengpu2.cluster.149333.0"}

    # Figure for losses and metrics
    fig1, axes1 = plt.subplots(nrows=2, ncols=2, figsize=(10, 10))
    fig2, axes2 = plt.subplots(nrows=3, ncols=2, figsize=(10, 14))

    for train_name, end_folder in train_names.items():
        case_folder = data_dir + train_name + end_folder
        event_acc = EventAccumulator(case_folder)
        event_acc.Reload()
        # Show all tags in the log file
        # print(event_acc.Tags())
        losses = ["InsideLoss", "LaplacianLoss"]
        metrics = ["residual", "l2_norm", "inf_norm"]
        data_types = ["train", "valid"]
        
        for i, loss in enumerate(losses):
            for j, data_type in enumerate(data_types):
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                _, epochs, vals = zip(*event_acc.Scalars(f'ComposedLosses/{loss}/{data_type}'))
                ax_prop(axes1[i][j], epochs, vals, train_name, f'{loss}/{data_type}')
        
        for i in range(2):
            for j in range(2):
                axes1[i][j].legend()
        
        for i, metric in enumerate(metrics):
            for j, data_type in enumerate(data_types):
                # E. g. get wall clock, number of steps and value for a scalar 'Accuracy'
                _, epochs, vals = zip(*event_acc.Scalars(f'Metrics/{metric}/{data_type}'))
                ax_prop(axes2[i][j], epochs, vals, train_name, f'{metric}/{data_type}')
        
        for i in range(3):
            for j in range(2):
                axes2[i][j].legend()

    fig1.savefig(fig_dir + "losses", bbox_inches='tight')
    fig2.savefig(fig_dir + "metrics", bbox_inches='tight')