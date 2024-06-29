import math
try:
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    from matplotlib.patches import FancyArrowPatch
except ImportError:
    print("Unable to import matplotlib. Cannot generate figures but neural net code itself can run")

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
    plt.style.use("fivethirtyeight")
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
    plt.style.use("fivethirtyeight")

    ax1: plt.Axes
    ax2: plt.Axes
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
    ax1:plt.Axes
    ax2:plt.Axes

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


def setup_bool_gradient(ax:plt.Axes, meshgrid, title=None):
    ax.set(xticks=[0, 1], yticks=[0, 1], xlim=(-0.5, 1.5), ylim=(-0.5, 1.5))
    ax.set_aspect('equal') # make sure it's square
    ax.set_title(title, fontsize=12)

    # Draw and label XOR input/outputs
    ax.scatter(*zip(*True_XOR), color=(0,0,0), label='True', s=100, marker='.')
    ax.scatter(*zip(*False_XOR), color=(0,0,0), label='False', s=80, marker='x')
    for (x, y) in True_XOR + False_XOR:
        ax.text(x+0.02, y-0.2, f'({x}, {y})', fontsize=8)

    # Draw the x and y axes
    # ax.axhline(0, color='white', linewidth=0.5, linestyle='-.')
    # ax.axvline(0, color='white', linewidth=0.5, linestyle='-.')
    # ax.axhline(0, color='black', linewidth=1, linestyle='dotted')
    # ax.axvline(0, color='black', linewidth=1, linestyle='dotted')

    # Overlay our activation values
    gradient = ax.imshow(
        meshgrid,
        extent=[-0.5, 1.5, -0.5, 1.5],
        origin='lower',     # data is x,y coordinates, not i,j row-columns
        cmap='coolwarm_r',  # red (0) to blue (1) gradient
        alpha=0.7           # make it lighter
    )
    return gradient


def plot_meshgrid(meshgrid, title):
    ax:plt.Axes
    fig, ax = plt.subplots(figsize=(4,3))

    color_grad = setup_bool_gradient(ax, meshgrid, title)
    cbar = plt.colorbar(color_grad)    # label="activation $a = \sigma(z)$")
    cbar.set_ticks([0, 0.25, 0.5, 0.75, 1])

    # plt.legend()
    plt.show()


def plot_net_meshgrids(node1grid, node2grid, outputgrid):
    plt.style.use('default')

    fig = plt.figure(figsize=(8, 6))
    gs = gridspec.GridSpec(4, 2, wspace=0.3, hspace=0.4)
    # (wspace = width between columns as a fraction of the average axis width)
    #gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 0.05])
    ax1 = fig.add_subplot(gs[0:2, 0])      # top-left plot
    ax2 = fig.add_subplot(gs[2:4, 0])      # bottom-left plot
    ax3 = fig.add_subplot(gs[1:3, 1])      # center-right plot

    ax1.set_ylabel('Node 1', fontsize=12) #, rotation=0)
    ax2.set_ylabel('Node 2', fontsize=12)

    setup_bool_gradient(ax1, node1grid, 'Layer 1')
    setup_bool_gradient(ax2, node2grid)
    color_grad = setup_bool_gradient(ax3, outputgrid, 'Output Node')

    # Create a single colorbar for all subplots
    cax = fig.add_subplot(gs[3, 1])
    cbar = plt.colorbar(color_grad, cax=cax, orientation='horizontal', ticks=[0, 1], label='classification')
    cbar.ax.xaxis.set_label_position('top')
    cbar.ax.set_position([0.61, 0.18, 0.25, 0.025])  # [left, bottom, width, height] in figure coordinates
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['False', 'True'])

    # create a single legend for all subplots
    handles, labels = ax3.get_legend_handles_labels()
    fig.legend(handles, labels, ncol=2, fontsize=8, title='XOR target classifications', title_fontsize=9,
            loc='lower right', bbox_to_anchor=(0.85, 0.06), bbox_transform=fig.transFigure)

    def middle_of_right_spline(ax:plt.Axes):
        bbox = ax.get_position()
        bbox.x1 -= 0.002 # empirical nudge to x value
        return [bbox.x1, (bbox.y0 + bbox.y1)/2]
    def middle_of_left_spline(ax:plt.Axes):
        bbox = ax.get_position()
        return [bbox.x0, (bbox.y0 + bbox.y1)/2]

    arrow_1_start = middle_of_right_spline(ax1)
    arrow_1_stop = middle_of_left_spline(ax3)
    arrow_2_start = middle_of_right_spline(ax2)
    arrow_2_stop = middle_of_left_spline(ax3)

    arrow1 = FancyArrowPatch(arrow_1_start, arrow_1_stop, linewidth=1.25, color='black', arrowstyle='->', 
                            connectionstyle="arc,angleA=5,angleB=180,armA=50,armB=75,rad=50", mutation_scale=20)
    arrow2 = FancyArrowPatch(arrow_2_start, arrow_2_stop, linewidth=1.25, color='black', arrowstyle='->', 
                            connectionstyle="arc,angleA=-5,angleB=180,armA=50,armB=75,rad=50", mutation_scale=20)

    fig.add_artist(arrow1)
    fig.add_artist(arrow2)

    plt.show()