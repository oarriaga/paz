import os
from cycler import cycler
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from matplotlib.ticker import AutoMinorLocator, AutoLocator
from matplotlib import rc
import arviz as az
import arviz.labels as azl
import jax
import jax.numpy as jp
import numpy as np
from tensorflow_probability.substrates import jax as tfp
import ternary
from ternary.helpers import simplex_iterator
tfd = tfp.distributions
tfb = tfp.bijectors


DANDELION = [0.992, 0.737, 0.258]
LABEL_COLOR = [0, 0, 0]
FONT = 'Palatino'
FONT = 'Times New Roman'
FONT_SIZE = 20
NAME_TO_TEX = {
    'shift': r'$x$-$y$ translation [m]',
    'x-translation': r'$x$ [m]',
    'y-translation': r'$y$ [m]',
    'translation_00': r'$x-y$ [m]',
    'theta_00': r'$\theta$ [rad]',
    'theta': r'$\theta$ [rad]',
    'sigma': r'$\sigma$',
    'noise': r'RGB pixel error',
    'scale': r'scale',
    'x-scale': r'scale $s_x$ [m]',
    'y-scale': r'scale $s_y$ [m]',
    'z-scale': r'scale $s_z$ [m]',
    'is_active': r'active',
    'color': r'color [rgb]',
    'ambient': r'ambient [$k_a$]',
    'diffuse': r'diffuse [$k_d$]',
    'specular': r'specular [$k_s$]',
    'shininess': r'shininess [$\alpha$]',
    'classes': r'classes',
    'class_0': r'$\kappa_{\mathrm{sphere}}$',
    'class_1': r'$\kappa_{\mathrm{cube}}$',
    'class_2': r'$\kappa_{\mathrm{cylinder}}$',
    'experimental-ambient': r'ambient [$k_a$]',
    'experimental-diffuse': r'diffuse [$k_d$]',
    'experimental-specular': r'specular [$k_s$]',
    'experimental-shininess': r'shininess [$\alpha$]',
    'R-color': r'color [r]',
    'G-color': r'color [g]',
    'B-color': r'color [b]',
    'experimental-R-color': r'color [r]',
    'experimental-G-color': r'color [g]',
    'experimental-B-color': r'color [b]',
}

rc('font', **{'family': 'serif', 'serif': [FONT], 'size': FONT_SIZE})
rc('text', usetex=True)

labeller = azl.MapLabeller(var_name_map={"x": r"x [m]", "y": r"y [m]"})

MARKER_SIZE = 20

point_estimate_kwargs = {
    'color': DANDELION,
    'linewidth': 1,
    'linestyle': (0, (5, 4))}

point_estimate_marker_kwargs = {
    'color': DANDELION,
    's': MARKER_SIZE,
    'label': 'posterior median',
    'zorder': 3}


kde_kwargs = {
    'hdi_probs': [0.05, 0.10, 0.20, 0.40, 0.60, 0.80],
    'contourf_kwargs': {'cmap': 'viridis'}}

pair_kwargs = {
    'kind': 'kde',
    'marginals': True,
    'point_estimate': 'median',
    'colorbar': False,
    'labeller': labeller,
    'point_estimate_kwargs': point_estimate_kwargs,
    'marginal_kwargs': {'color': DANDELION},
    'kde_kwargs': kde_kwargs,
    'point_estimate_marker_kwargs': point_estimate_marker_kwargs}


posterior_kwargs = {'outline': True,
                    'shade': 0.4,
                    'point_estimate': 'mean',
                    'hdi_prob': 1.0,
                    'bw': 'scott'}


corner_A = [0, 0]
corner_B = [1, 0]
corner_C = [0.5, 0.75**0.5]
TRIANGLE_AREA = (1 * (0.75**0.5)) / 2.0  # base * height / 2.0
TRIANGLE_CORNERS = np.array([corner_A, corner_B, corner_C])
triangle = tri.Triangulation(TRIANGLE_CORNERS[:, 0], TRIANGLE_CORNERS[:, 1])
pairs = [np.array([[corner_B], [corner_C]]),
         np.array([[corner_C], [corner_A]]),
         np.array([[corner_A], [corner_B]])]


def has_extension(fullpath, extension):
    return fullpath.endswith(extension)


def validate_extension(filename, extension):
    if not has_extension(filename, extension):
        raise ValueError(f'Filename {filename} missing extension {extension}')


def write_image(image, directory, filename):
    validate_extension(filename, '.png')
    filepath = os.path.join(directory, filename)
    plt.imsave(filepath, image)


def write_losses(losses, directory, filename):
    figure, axis = plt.subplots()
    axis.plot(losses, color=DANDELION)
    axis.set_ylabel('loss')
    axis.set_xlabel('step')
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    fullpath = os.path.join(directory, filename)
    validate_extension(fullpath, '.pdf')
    figure.savefig(fullpath, bbox_inches='tight')
    plt.close()


def plot_forward_samples(forward_samples, name, directory):
    figure, axis = plt.subplots()
    az.plot_dist(forward_samples, color=DANDELION, ax=axis,
                 plot_kwargs={'linewidth': 5}, fill_kwargs={'alpha': 0.4})
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.xaxis.labelpad = 20
    axis.yaxis.labelpad = 20

    axis.set_ylabel('density')
    axis.set_xlabel(NAME_TO_TEX[name])

    filename = os.path.join(directory, f'prior_forward_{name}.pdf')
    figure.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_inverse_samples(inverse_samples, normals_samples, name, directory):
    figure, axis = plt.subplots()
    kwargs = {'plot_kwargs': {'linewidth': 5}, 'fill_kwargs': {'alpha': 0.4}}
    az.plot_dist(normals_samples, color='r', ax=axis, label='normal', **kwargs)
    az.plot_dist(inverse_samples, color='b', ax=axis,
                 label='inverse', **kwargs)
    axis.spines['top'].set_visible(False)
    axis.spines['right'].set_visible(False)
    axis.spines['left'].set_visible(False)
    axis.xaxis.labelpad = 20
    axis.yaxis.labelpad = 20
    axis.legend(prop={'size': 10}, frameon=False)
    axis.set_ylabel('density')
    axis.set_xlabel(NAME_TO_TEX[name])
    filename = os.path.join(directory, f'prior_inverse_{name}.pdf')
    figure.savefig(filename, bbox_inches="tight")
    plt.close()


def plot_priors(key, variables, directory, num_samples=10_000):
    keys = jax.random.split(key, len(variables))
    for key, variable in zip(keys, variables):
        plot_prior(key, variable, directory, num_samples)


def plot_forward_dirichlet(forward_samples, directory, num_samples=10_000,
                           scale=30, cmap='viridis'):

    figure, axis = plt.subplots()
    corners = list(simplex_iterator(scale=scale))
    corners = np.array(corners) / scale

    corner_density = compute_corner_density(corners, forward_samples)
    density = wrap_density(scale, corner_density)
    ternary.heatmap(density, scale, ax=axis, cmap=cmap,
                    colorbar=False, style='h')
    plot_dirichlet_colorbar(axis, density, cmap)
    plot_dirichlet_ticks(axis, scale, fontsize=10)
    point_estimate = np.mean(forward_samples, axis=0) * scale
    plot_dirichlet_point_estimate(axis, scale, point_estimate, DANDELION)
    plot_dirichlet_labels(axis, scale)
    filepath = os.path.join(directory, 'dirichlet_prior.pdf')
    figure.savefig(filepath, bbox_inches='tight')


def plot_prior(key, variable, directory, num_samples=10_000):
    key_0, key_1 = jax.random.split(key)
    forward_samples = variable.sample(key_0, num_samples)
    inverse_samples = variable.sample_inverse(key_1, num_samples)
    normals_samples = tfd.Normal(0.0, 1.00).sample(num_samples, seed=key_1)
    name = variable.name
    if name == 'shift':
        for arg, shift_name in enumerate(['x-translation', 'y-translation']):
            plot_forward_samples(forward_samples[:, arg],
                                 shift_name, directory)
            plot_inverse_samples(inverse_samples[:, arg],
                                 normals_samples, shift_name, directory)
    elif name == 'color':
        for arg, channel_name in enumerate(['R-color', 'G-color', 'B-color']):
            plot_forward_samples(forward_samples[:, arg],
                                 channel_name, directory)
            plot_inverse_samples(inverse_samples[:, arg],
                                 normals_samples, channel_name, directory)
    elif name == 'classes':
        plot_forward_dirichlet(forward_samples, directory, num_samples)
        plot_inverse_samples(inverse_samples, normals_samples, name, directory)
    elif name == 'experimental-color':
        for arg, channel_name in enumerate(['experimental-R-color',
                                            'experimental-G-color',
                                            'experimental-B-color']):
            plot_forward_samples(forward_samples[:, arg],
                                 channel_name, directory)
            plot_inverse_samples(inverse_samples[:, arg],
                                 normals_samples, channel_name, directory)
    else:
        plot_forward_samples(forward_samples, name, directory)
        plot_inverse_samples(inverse_samples, normals_samples, name, directory)


def mode(values):
    return az.plots.plot_utils.calculate_point_estimate('mode', values)


def median(values):
    return az.plots.plot_utils.calculate_point_estimate('median', values)


def build_summary(trace):
    variable_names = list(trace.keys())
    summary = az.summary(trace, var_names=variable_names,
                         stat_funcs=[mode, median])
    return summary


def _parse_summary(summary, statistic='mean'):
    x_shift = summary[statistic]['shift[0]']
    y_shift = summary[statistic]['shift[1]']
    shift = jp.array([x_shift, y_shift])

    theta = jp.array(summary['mode']['theta'])

    x_scale = summary[statistic]['scale[0]']
    y_scale = summary[statistic]['scale[1]']
    z_scale = summary[statistic]['scale[2]']
    scale = jp.array([x_scale, y_scale, z_scale])

    r_color = summary[statistic]['color[0]']
    g_color = summary[statistic]['color[1]']
    b_color = summary[statistic]['color[2]']
    color = jp.array([r_color, g_color, b_color])

    ambient = jp.array(summary[statistic]['ambient'])
    diffuse = jp.array(summary[statistic]['diffuse'])
    specular = jp.array(summary[statistic]['specular'])
    shininess = jp.array(summary[statistic]['shininess'])

    class_0 = summary[statistic]['classes[0]']
    class_1 = summary[statistic]['classes[1]']
    class_2 = summary[statistic]['classes[2]']
    classes = jp.array([class_0, class_1, class_2])
    return {'shift': shift, 'theta': theta, 'scale': scale,
            'color': color, 'ambient': ambient, 'diffuse': diffuse,
            'specular': specular, 'shininess': shininess, 'classes': classes}


def write_point_estimate(summary, render, directory, statistic='mean'):
    sample = _parse_summary(summary, statistic)
    point_image, point_depth = render(sample)
    write_image(point_image, directory, 'mean_point_image.png')
    write_image(point_depth, directory, 'mean_point_depth.png')
    return point_image, point_depth


def write_true_pred_image(true_image, pred_image, directory, label):
    image = jp.concatenate([true_image, pred_image], axis=1)
    write_image(image, directory, f'true_pred_{label}_image.png')


def write_error_image(true_image, pred_image, directory):
    image = (true_image - pred_image)**2
    write_image(image, directory, 'error_image.png')


def compute_limits(samples, true_mean, half_size):
    data_max = samples.max()
    data_min = samples.min()
    true_min = true_mean - half_size
    true_max = true_mean + half_size
    limit_min = np.max([true_min, data_min])
    limit_max = np.min([true_max, data_max])
    return [limit_min, limit_max]


def plot_shift_posterior(trace, label, directory, name='shift'):
    x_samples = trace[name][:, :, 0]
    y_samples = trace[name][:, :, 1]
    xz_trace = {'x': x_samples, 'y': y_samples}
    axes = az.plot_pair(xz_trace, var_names=['x', 'y'], **pair_kwargs)
    half_box = 0.1 / 2.0
    summary = az.summary(xz_trace, stat_funcs=[mode, median])
    x_mean = summary['median'].x
    y_mean = summary['median'].y
    x_mean_round = round(x_mean, 3)
    y_mean_round = round(y_mean, 3)
    x_limits = compute_limits(x_samples.flatten(), x_mean_round, half_box)
    y_limits = compute_limits(y_samples.flatten(), y_mean_round, half_box)
    axes[1, 0].set_xlim(x_limits)
    axes[1, 0].set_ylim(y_limits)
    # plot label
    y_true = list(label.values())[0]['x'] * -1
    x_true = list(label.values())[0]['y']
    axes[1, 0].scatter(x_true, y_true, s=MARKER_SIZE,
                       marker=(5, 1), c='red', label='true point', zorder=3)

    axes[1, 0].legend(prop={'size': 10}, frameon=False)
    axes[1, 0].spines["top"].set_visible(False)
    axes[1, 0].spines["right"].set_visible(False)
    axes[1, 0].spines["left"].set_visible(False)
    axes[1, 0].spines["bottom"].set_visible(False)
    axes[1, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[1, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[1, 1].axis('off')
    axes[0, 0].axis('off')
    plt.savefig(os.path.join(directory, f'{name}_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_theta_posterior(trace, directory, bins=100, label=None):
    x = np.array(trace['theta'].flatten())
    figure, axis = plt.subplots(1, 1, subplot_kw=dict(projection='polar'))
    x = (x + np.pi) % (2 * np.pi) - np.pi  # Wrap angles to [-pi, pi)
    count, bin_edges = np.histogram(x, bins=bins)

    area = count / x.size  # Area to assign each bin
    count = (area / np.pi) ** .5  # Calculate corresponding bin radius

    bin_widths = np.diff(bin_edges)
    axis.bar(bin_edges[:-1], count, zorder=1, align='edge', width=bin_widths,
             color=DANDELION, edgecolor=DANDELION, fill=True, linewidth=1)

    label_position = axis.get_rlabel_position()
    label_position = np.radians(label_position + 10)
    location = 3.1415 / 2.0
    axis.text(location, axis.get_rmax() * 0.75, 'density', rotation=0.0,
              ha='center', va='center')
    # axis.set_ylabel('density', rotation=45)
    if label is not None:
        true_angle = list(label.values())[0]['theta_z']
        axis.vlines(true_angle, 0, count.max(), colors=LABEL_COLOR,
                    zorder=3, linewidth=3)

    plt.savefig(os.path.join(directory, 'theta_posterior.pdf'))
    plt.close()


def get_default_RGB():
    EARTH_R = '#d62728'
    JADED_G = '#2ca02c'
    LIGHT_B = '#1f77b4'
    return EARTH_R, JADED_G, LIGHT_B


def configure_shift(kwargs):
    color_2, color_3, color_1 = get_default_RGB()
    rc('axes', prop_cycle=cycler(color=[color_1, color_2, color_3]))
    kwargs['labelcolor'] = [color_1, color_2, color_3]
    kwargs['ncol'] = 2
    legend = [r'\textbf{x}', r'\textbf{y}']
    return kwargs, legend


def configure_scale(kwargs):
    color_1, color_2, color_3 = get_default_RGB()
    rc('axes', prop_cycle=cycler(color=[color_1, color_2, color_3]))
    kwargs['labelcolor'] = [color_1, color_2, color_3]
    kwargs['ncol'] = 3
    legend = [r'\textbf{x}', r'\textbf{y}', r'\textbf{z}']
    return kwargs, legend


def configure_color(kwargs):
    color_1, color_2, color_3 = get_default_RGB()
    rc('axes', prop_cycle=cycler(color=[color_1, color_2, color_3]))
    kwargs['labelcolor'] = [color_1, color_2, color_3]
    kwargs['ncol'] = 3
    legend = [r'\textbf{r}', r'\textbf{g}', r'\textbf{b}']
    return kwargs, legend


def configure_shape(kwargs):
    color_1, color_2, color_3 = get_default_RGB()
    rc('axes', prop_cycle=cycler(color=[color_1, color_2, color_3]))
    kwargs['ncol'] = 3
    kwargs['labelcolor'] = [color_1, color_2, color_3]
    # legend = [r'\textbf{sphere}', r'\textbf{cube}', r'\textbf{cylinder}']
    legend = [NAME_TO_TEX['class_0'],
              NAME_TO_TEX['class_1'],
              NAME_TO_TEX['class_2']]
    return kwargs, legend


def configure_default(kwargs):
    color_2, color_3, color_1 = get_default_RGB()
    rc('axes', prop_cycle=cycler(color=[color_1, color_2, color_3]))
    return None, None


def name_to_legend_kwargs(name):
    kwargs = {'prop': {'size': 10}, 'frameon': False, 'handletextpad': 0.0,
              'handlelength': 0, 'columnspacing': 0.45, 'loc': 'upper right'}
    if name == 'shift':
        kwargs, legend = configure_shift(kwargs)
    elif name == 'scale':
        kwargs, legend = configure_scale(kwargs)
    elif name == 'color':
        kwargs, legend = configure_color(kwargs)
    elif name == 'classes':
        kwargs, legend = configure_shape(kwargs)
    else:
        kwargs, legend = configure_default(kwargs)
    return legend, kwargs


def plot_trace_variable(trace, variable_name, directory):
    legend, legend_kwargs = name_to_legend_kwargs(variable_name)
    axes = az.plot_trace(trace, var_names=variable_name)
    axes[0, 0].set_xlabel(NAME_TO_TEX[variable_name])
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['left'].set_visible(False)
    axes[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0, 0].get_yaxis().set_major_locator(AutoLocator())
    axes[0, 0].xaxis.labelpad = 10
    axes[0, 0].yaxis.labelpad = 10
    axes[0, 0].set_ylabel('frequency')
    axes[0, 0].set_title('')
    if legend is not None:
        axes[0, 0].legend(legend, **legend_kwargs)

    axes[0, 1].spines['top'].set_visible(False)
    axes[0, 1].spines['right'].set_visible(False)
    axes[0, 1].spines['left'].set_visible(False)
    axes[0, 1].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0, 1].xaxis.labelpad = 10
    axes[0, 1].yaxis.labelpad = 10
    axes[0, 1].set_ylabel(NAME_TO_TEX[variable_name])
    axes[0, 1].set_xlabel('sample iteration')
    axes[0, 1].set_title('')
    if legend is not None:
        axes[0, 1].legend(legend, **legend_kwargs)

    filepath = os.path.join(directory, f'trace_{variable_name}.pdf')
    plt.savefig(filepath, bbox_inches='tight')
    plt.close()


def plot_trace(trace, directory):
    plt.rcParams.update({'font.size': 12})
    for variable_name in list(trace.keys()):
        plot_trace_variable(trace, variable_name, directory)

    az.plot_trace(trace)
    filepath = os.path.join(directory, 'trace.pdf')
    plt.savefig(filepath)
    plt.close()

    plt.rcParams.update({'font.size': FONT_SIZE})


def prettify_posterior_plot(axes, variable_name):
    axes[0, 0].legend(prop={'size': 10}, frameon=False)
    axes[0, 0].spines['top'].set_visible(False)
    axes[0, 0].spines['right'].set_visible(False)
    axes[0, 0].spines['left'].set_visible(False)
    axes[0, 0].xaxis.labelpad = 20
    axes[0, 0].yaxis.labelpad = 20
    axes[0, 0].set_ylabel('density')
    axes[0, 0].set_xlabel(NAME_TO_TEX[variable_name])
    axes[0, 0].xaxis.set_minor_locator(AutoMinorLocator())
    axes[0, 0].yaxis.set_minor_locator(AutoMinorLocator())
    axes[0, 0].yaxis.set_major_locator(AutoLocator())
    axes[0, 0].set_title('')
    return axes


def prettify_legend(axes, text, colors, num_cols):
    kwargs = {'prop': {'size': 10},
              'frameon': False,
              'handletextpad': 0.0,
              'handlelength': 0,
              'columnspacing': 0.45,
              'loc': 'upper right',
              'labelcolor': colors,
              'ncol': num_cols}
    axes[0, 0].legend(text, **kwargs)
    return axes


def plot_shift_posteriors(trace, directory, kwargs=posterior_kwargs):
    x = az.convert_to_dataset(np.array(trace['shift'][:, :, 0]))
    y = az.convert_to_dataset(np.array(trace['shift'][:, :, 1]))
    kwargs['colors'] = get_default_RGB()
    axes = az.plot_density([x, y], **kwargs)
    axes = prettify_posterior_plot(axes, 'shift')
    text = [r'\textbf{x}', r'\textbf{y}']
    prettify_legend(axes, text, kwargs['colors'], 2)
    plt.savefig(os.path.join(directory, 'shift_posteriors.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_shift_x_posterior(trace, directory, kwargs=posterior_kwargs):
    x = az.convert_to_dataset(np.array(trace['shift'][:, :, 0]))
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density([x], **kwargs)
    axes = prettify_posterior_plot(axes, 'x-translation')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'shift-x_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_shift_y_posterior(trace, directory, kwargs=posterior_kwargs):
    y = az.convert_to_dataset(np.array(trace['shift'][:, :, 1]))
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density([y], **kwargs)
    axes = prettify_posterior_plot(axes, 'y-translation')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'shift-y_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_color_posterior(trace, directory, kwargs=posterior_kwargs):
    r = az.convert_to_dataset(np.array(trace['color'][:, :, 0]))
    g = az.convert_to_dataset(np.array(trace['color'][:, :, 1]))
    b = az.convert_to_dataset(np.array(trace['color'][:, :, 2]))
    kwargs['colors'] = get_default_RGB()
    axes = az.plot_density([r, g, b], **kwargs)
    prettify_posterior_plot(axes, 'color')
    text = [r'\textbf{r}', r'\textbf{g}', r'\textbf{b}']
    prettify_legend(axes, text, kwargs['colors'], 3)
    plt.savefig(os.path.join(directory, 'color_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_scale_posterior(trace, directory, kwargs=posterior_kwargs):
    x = az.convert_to_dataset(np.array(trace['scale'][:, :, 0]))
    y = az.convert_to_dataset(np.array(trace['scale'][:, :, 1]))
    z = az.convert_to_dataset(np.array(trace['scale'][:, :, 2]))
    kwargs['colors'] = get_default_RGB()
    axes = az.plot_density([x, y, z], **kwargs)
    prettify_posterior_plot(axes, 'scale')
    text = [r'$s_x$', r'$s_y$', r'$s_z$']
    prettify_legend(axes, text, kwargs['colors'], 3)
    plt.savefig(os.path.join(directory, 'scale_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_scale_x_posterior(trace, directory, kwargs=posterior_kwargs):
    x = az.convert_to_dataset(np.array(trace['scale'][:, :, 0]))
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density([x], **kwargs)
    prettify_posterior_plot(axes, 'x-scale')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'scale-x_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_scale_y_posterior(trace, directory, kwargs=posterior_kwargs):
    y = az.convert_to_dataset(np.array(trace['scale'][:, :, 1]))
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density([y], **kwargs)
    prettify_posterior_plot(axes, 'y-scale')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'scale-y_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_scale_z_posterior(trace, directory, kwargs=posterior_kwargs):
    z = az.convert_to_dataset(np.array(trace['scale'][:, :, 2]))
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density([z], **kwargs)
    prettify_posterior_plot(axes, 'z-scale')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'scale-z_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_classes_posterior(trace, directory, kwargs=posterior_kwargs):
    class_1 = az.convert_to_dataset(np.array(trace['classes'][:, :, 0]))
    class_2 = az.convert_to_dataset(np.array(trace['classes'][:, :, 1]))
    class_3 = az.convert_to_dataset(np.array(trace['classes'][:, :, 2]))
    kwargs['colors'] = get_default_RGB()
    axes = az.plot_density([class_1, class_2, class_3], **kwargs)
    prettify_posterior_plot(axes, 'classes')
    # text = [r'\textbf{sphere}', r'\textbf{cube}', r'\textbf{cylinder}']
    text = [NAME_TO_TEX['class_0'],
            NAME_TO_TEX['class_1'],
            NAME_TO_TEX['class_2']]
    prettify_legend(axes, text, kwargs['colors'], 3)
    plt.savefig(os.path.join(directory, 'classes_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_ambient_posterior(trace, directory, kwargs=posterior_kwargs):
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density(trace, var_names=['ambient'], **kwargs)
    prettify_posterior_plot(axes, 'ambient')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'ambient_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_diffuse_posterior(trace, directory, kwargs=posterior_kwargs):
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density(trace, var_names=['diffuse'], **kwargs)
    prettify_posterior_plot(axes, 'diffuse')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'diffuse_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_specular_posterior(trace, directory, kwargs=posterior_kwargs):
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density(trace, var_names=['specular'], **kwargs)
    prettify_posterior_plot(axes, 'specular')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'specular_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def plot_shininess_posterior(trace, directory, kwargs=posterior_kwargs):
    kwargs['colors'] = [DANDELION]
    axes = az.plot_density(trace, var_names=['shininess'], **kwargs)
    prettify_posterior_plot(axes, 'shininess')
    prettify_legend(axes, '', kwargs['colors'], 1)
    plt.savefig(os.path.join(directory, 'shininess_posterior.pdf'),
                bbox_inches='tight')
    plt.close()


def compute_pairwise_distances(x, y):
    """Compute euclidean distance for each vector x with each vector y

    # Arguments:
        x: Tensor with shape `(n, vector_dim)`
        y: Tensor with shape `(m, vector_dim)`

    # Returns:
        Tensor with shape `(n, m)` where each value pair n, m corresponds to
        the distance between the vector `n` of `x` with the vector `m` of `y`
    """
    n = x.shape[0]
    m = y.shape[0]
    x = np.tile(np.expand_dims(x, 1), [1, m, 1])
    y = np.tile(np.expand_dims(y, 0), [n, 1, 1])
    return np.mean(np.power(x - y, 2), 2)


def wrap_density(scale, corner_density):
    d = dict()
    for (i, j, k), density in zip(simplex_iterator(scale), corner_density):
        d[(i, j)] = density
    return d


def compute_corner_density(corners, probabilities):
    distances = compute_pairwise_distances(corners, probabilities)
    closest_corners = np.argmin(distances, axis=0)
    corner_density = []
    for corner_arg in range(len(corners)):
        density = np.sum(corner_arg == closest_corners)
        corner_density.append(density)
    return corner_density


def flatten_trace_variable(trace, variable='classes'):
    return trace[variable].reshape(-1, 3)


def plot_dirichlet_colorbar(axis, density, cmap, shrink=0.80):
    colorbar_kwargs = {'shrink': shrink, 'format': '%.0e'}
    vmin = min(density.values())
    vmax = max(density.values())
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    colorbar = plt.colorbar(sm, ax=axis, **colorbar_kwargs)
    colorbar.ax.get_yaxis().labelpad = 15
    colorbar.set_label('frequency', rotation=90, fontsize=15)
    colorbar.ax.tick_params(labelsize=10)
    return colorbar


def plot_dirichlet_ticks(axis, scale, fontsize=10):
    ticks = np.linspace(0, 1, 11).tolist()
    ticks_locations = np.linspace(0, scale, 11).tolist()
    ternary.lines.ticks(axis, scale, ticks=ticks, locations=ticks_locations,
                        clockwise=False, axis='blr', tick_formats=' %.1f',
                        offset=0.0180, fontsize=fontsize)
    ternary.plotting.clear_matplotlib_ticks(ax=axis, axis='both')


def plot_dirichlet_point_estimate(axis, scale, point_estimate, color):
    point_A = np.array([0, scale - point_estimate[0], point_estimate[0]])
    point_B = np.array([scale - point_estimate[1], point_estimate[1], 0])
    point_C = np.array([point_estimate[2], 0, scale - point_estimate[2]])

    ternary.line(axis, point_A, point_estimate, '012', color=color)
    ternary.line(axis, point_B, point_estimate, '012', color=color)
    ternary.line(axis, point_C, point_estimate, '012', color=color)

    x, y, z = point_estimate / scale
    point_estimate = np.reshape(point_estimate, (1, 3))
    ternary.scatter(points=point_estimate, ax=axis, marker='o', color=color,
                    label=f'mean point: [{x:.2f}, {y:.2f}, {z:.2f}]')
    axis.legend(prop={'size': 10}, loc=(0.6, 0.95), frameon=False)
    axis.axis('off')


def plot_dirichlet_labels(axis, scale):
    figure, tax = ternary.figure(axis, scale=scale)
    tax.left_axis_label(NAME_TO_TEX['class_0'], offset=0.18)
    tax.right_axis_label(NAME_TO_TEX['class_1'], offset=0.18)
    tax.bottom_axis_label(NAME_TO_TEX['class_2'], offset=0.1)


def plot_dirichlet_posterior(trace, directory, scale=20, cmap='viridis'):
    figure, axis = plt.subplots()
    corners = list(simplex_iterator(scale=scale))
    corners = np.array(corners) / scale
    probabilities = flatten_trace_variable(trace, 'classes')

    probabilities = trace['classes'].reshape(-1, 3)
    corner_density = compute_corner_density(corners, probabilities)
    density = wrap_density(scale, corner_density)
    ternary.heatmap(density, scale, ax=axis, cmap=cmap,
                    colorbar=False, style='h')
    plot_dirichlet_colorbar(axis, density, cmap)
    plot_dirichlet_ticks(axis, scale, fontsize=10)
    point_estimate = np.mean(probabilities, axis=0) * scale
    plot_dirichlet_point_estimate(axis, scale, point_estimate, DANDELION)
    plot_dirichlet_labels(axis, scale)
    filepath = os.path.join(directory, 'dirichlet_posterior.pdf')
    figure.savefig(filepath, bbox_inches='tight')
