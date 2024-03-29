# Generated by Django 4.0.1 on 2022-03-26 12:15

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('alpha_business_app', '0008_agentconfig_and_more'),
    ]

    operations = [
        migrations.RenameField(
            model_name='container',
            old_name='config_file',
            new_name='config',
        ),
        migrations.AlterField(
            model_name='agentconfig',
            name='agents_config',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='alpha_business_app.agentsconfig'),
        ),
        migrations.AlterField(
            model_name='config',
            name='environment',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='alpha_business_app.environmentconfig'),
        ),
        migrations.AlterField(
            model_name='config',
            name='hyperparameter',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='alpha_business_app.hyperparameterconfig'),
        ),
        migrations.AlterField(
            model_name='environmentconfig',
            name='agents',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='alpha_business_app.agentsconfig'),
        ),
        migrations.AlterField(
            model_name='hyperparameterconfig',
            name='rl',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='alpha_business_app.rlconfig'),
        ),
        migrations.AlterField(
            model_name='hyperparameterconfig',
            name='sim_market',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='alpha_business_app.simmarketconfig'),
        ),
        migrations.DeleteModel(
            name='CERebuyAgentQLearningConfig',
        ),
    ]
