# Generated by Django 4.0.1 on 2022-03-22 18:41

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('alpha_business_app', '0007_rename_agent_environmentconfig_agents_and_more'),
    ]

    operations = [
        migrations.CreateModel(
            name='AgentConfig',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(default='', max_length=100)),
                ('agent_class', models.CharField(max_length=100, null=True)),
                ('argument', models.CharField(max_length=200, null=True)),
                ('agents_config', models.ForeignKey(null=True, on_delete=django.db.models.deletion.DO_NOTHING, to='alpha_business_app.agentsconfig')),
            ],
        ),
        migrations.RenameField(
            model_name='simmarketconfig',
            old_name='episode_size',
            new_name='episode_length',
        ),
        migrations.AddField(
            model_name='config',
            name='name',
            field=models.CharField(default='', editable=False, max_length=100),
        ),
        migrations.AlterField(
            model_name='container',
            name='config_file',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='alpha_business_app.config'),
        ),
        migrations.DeleteModel(
            name='RuleBasedAgentConfig',
        ),
    ]