# Generated by Django 4.0.1 on 2022-03-19 15:57

import django.db.models.deletion
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('alpha_business_app', '0007_rename_agent_environmentconfig_agents_and_more'),
    ]

    operations = [
        migrations.AlterField(
            model_name='container',
            name='config_file',
            field=models.ForeignKey(on_delete=django.db.models.deletion.DO_NOTHING, to='alpha_business_app.config'),
        ),
    ]
