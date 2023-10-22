# src/visualization/visualize.py
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import click
import logging
from pathlib import Path

@click.command()
@click.argument('input_filepath', type=click.Path(exists=True))
@click.argument('output_filepath', type=click.Path())
def main(input_filepath, output_filepath):
    """ Runs scripts to create visualizations from processed data
        saved in input_filepath and saves them in output_filepath.
    """
    logger = logging.getLogger(__name__)
    logger.info('Creating visualizations')

    amex_train_fig = pd.read_csv(input_filepath)
    amex_train_fig = amex_train_fig.drop('Unnamed: 0', axis=1)

    # Distribution of Credit Card Default Customers
    create_distribution_plot(amex_train_fig, output_filepath)

    # Heatmap Matrix Correlation
    create_heatmap(amex_train_fig, output_filepath)

    # Histogram plots
    create_histograms(amex_train_fig, output_filepath)

    # Yearly Debt Payments Distribution By Default History
    create_debt_distribution_plot(amex_train_fig, output_filepath)

    # Net Yearly Income By Occupation Type
    create_income_occupation_plot(amex_train_fig, output_filepath)

    # Credit Score By Customers Default History
    create_credit_score_boxplot(amex_train_fig, output_filepath)


def create_distribution_plot(df, output_filepath):
    sns.countplot(data=df, x='credit_card_default', palette='Set2')
    plt.title('Distribution of Credit Card Default Customers')
    plt.xlabel('Credit Card Default')
    plt.ylabel('Frequency')
    plt.xticks(ticks=[0, 1], labels=['No Default', 'Default'])
    plt.grid(False)
    plt.savefig(f'{output_filepath}/distribution_ccd.png')
    plt.show()


def create_heatmap(df, output_filepath):
    plt.figure(figsize=(15, 8))
    sns.heatmap(df.corr(numeric_only=True), annot=True)
    plt.title('Heatmap Matrix Correlation American Express Dataframe')
    plt.savefig(f'{output_filepath}/heatmap.png', bbox_inches='tight')
    plt.show()


def create_histograms(df, output_filepath):
    sns.set_palette('Set2')
    ax = df.hist(figsize=(15, 10))
    plt.subplots_adjust(hspace=0.5)
    for axis in ax.flatten():
        axis.grid(False)
    plt.savefig(f'{output_filepath}/histogram_plots.png')
    plt.show()


def create_debt_distribution_plot(df, output_filepath):
    sns.histplot(data=df, x='yearly_debt_payments', hue='credit_card_default', palette='Set2')
    plt.grid(False)
    plt.title('Yearly Debt Payments Distribution By Default History')
    plt.savefig(f'{output_filepath}/distribution_payments.png')
    plt.show()


def create_income_occupation_plot(df, output_filepath):
    g = sns.catplot(x='occupation_type',
                    data=df,
                    y='net_yearly_income',
                    hue='credit_card_default',
                    kind='bar',
                    palette='Set2',
                    errorbar=None
                    )
    g.set_xticklabels(rotation=90, horizontalalignment='right')
    plt.title('Net Yearly Income By Occupation Type')
    g.savefig(f'{output_filepath}/income_occupation.png')
    plt.show()


def create_credit_score_boxplot(df, output_filepath):
    sns.boxplot(x='credit_card_default', y='credit_score', data=df, palette='Set2')
    plt.title('Credit Score By Customers Default History')
    plt.savefig(f'{output_filepath}/boxplot_creditscore.png')
    plt.show()


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # Not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    main()
