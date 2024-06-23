import math
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    print("Unable to import matplotlib. Cannot generate figures but neural net code itself can run")

plt.style.use("fivethirtyeight")

True_XOR = [(0,1), (1,0)]
False_XOR = [(0,0), (1,1)]
True_OR = [(0,1), (1,0), (1,1)]
False_OR = [(0,0)]

def setup_bool(ax:plt.Axes, true_points, false_points, label_points=True):
    ax.set(xticks=[0, 1], yticks=[0, 1], xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    ax.set_aspect('equal') # make sure it's square

    # Draw and label Boolean input space
    ax.scatter(*zip(*true_points), c='blue', label='True', s=100)
    ax.scatter(*zip(*false_points), c='red', label='False', s=100)

    if label_points:
        for (x, y) in true_points + false_points:
            ax.text(x+0.04, y-0.1, f'({x}, {y})', fontsize=7)

    # Draw box around the Axes
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # Show legend per Axes
    # legend = ax.legend(loc='lower right', title='output', ncols=2, fontsize=7, title_fontsize=7)


def plot_binary_Boolean_input_space_with_XOR_output():
    '''plot binary Boolean function input space, with colors indicating the output of XOR'''
    fig = plt.figure(figsize=(3.5,2.5))
    ax = fig.add_subplot()
    setup_bool(ax, True_XOR, False_XOR, label_points=True)
    ax.set_title('XOR', y=1.05, fontsize=14)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=8, title='output values', title_fontsize=9,
            loc='lower center', bbox_to_anchor=(0.5, -0.25))
    plt.show()


def draw_boundary(ax:plt.Axes, y_int, slope,
                  label_x:float, label:str='Decision boundary', fontsize=9,
                  linestyle='--', linewidth='1.5'):
    '''Draw and label a decision boundary, with label text starting at x=label_x'''
    ax.axline((0, y_int), slope=slope, c='black', ls=linestyle, linewidth=linewidth)

    label_y = slope * label_x + y_int              # y = mx+b
    text_angle_rad = math.atan(slope)
    text_angle_deg = math.degrees(text_angle_rad)  # only works if the plot is square (set aspect='equal')

    # Desired offset distance
    offset_distance = 0.06

    # Calculate the offset coordinates
    dx = offset_distance * math.cos(text_angle_rad + math.pi / 2)
    dy = offset_distance * math.sin(text_angle_rad + math.pi / 2)

    if text_angle_deg > 0:
        ax.text(label_x+dx, label_y+dy, label, rotation=text_angle_deg, fontsize=fontsize, c='black', verticalalignment='bottom')
    else:
        ax.text(label_x+dx, label_y+dy, label, rotation=text_angle_deg, fontsize=fontsize, c='black', verticalalignment='top')


def plot_linear_separation_OR_vs_XOR():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize = (6.5, 3))
    fig.subplots_adjust(wspace=0.5)

    setup_bool(ax1, True_OR, False_OR)
    ax1.set_title('OR classification', fontsize=12)
    ax1.set_xlabel('x')
    ax1.set_ylabel('y', rotation=0, labelpad=10)
    draw_boundary(ax1, y_int=0.5, slope=-1, label_x=-0.2, label='Linear decision boundary')

    setup_bool(ax2, True_XOR, False_XOR)
    ax2.set_title('XOR classification...?', fontsize=12)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y', rotation=0, labelpad=10)
    ax2.plot([-0.5, 0.6], [0.1, 0.6], c='black', ls='--', lw=1.5)
    ax2.text(0.6, 0.6-0.025, s='??', rotation=25, fontsize=12, c='black')

    # Shared legend for figure; copy from Axes 1
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 0), ncol=2,
            fontsize=8, title='output values', title_fontsize=9)
    plt.show()


def plot_functions(sigmoid, sigmoid_derivative, log_loss):
    plt.style.use("default")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3))
    fig.subplots_adjust(wspace=0.3)

    x1 = [-5 + 0.25*i for i in range(40)]
    s = [sigmoid(n) for n in x1]
    sd = [sigmoid_derivative(n) for n in x1]

    ax1.set_title('sigmoid(z) and derivative', fontsize=10)
    ax1.plot(x1, s, label=r'$\sigma(z) = \frac{1}{1 + e^{-z}}$', color='blue')
    ax1.plot(x1, sd, label=r"$\sigma'(z) = \sigma(z) (1 - \sigma(z))$", color='red', zorder=0) # plot below s
    ax1.set_xlabel('z [pre-activation value]')
    ax1.set_ylabel('$a = \sigma(z)$')

    x2 = [0.01*(i) for i in range(100)]
    ll = [log_loss(0, n) for n in x2]

    ax2.set_title('log_loss when target_output=0', fontsize=10)
    ax2.plot(x2, ll, label='$log\_loss(0,a) = -log(1-a)$', color='blue')
    ax2.set_xlabel('$a = \sigma(z)$ [node activation]')
    ax2.set_ylabel('Loss')

    for ax in (ax1, ax2):
        ax.grid(True)
        ax.legend(fontsize=9)

    plt.show()