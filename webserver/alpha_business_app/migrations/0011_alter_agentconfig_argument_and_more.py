# Generated by Django 4.0.1 on 2022-04-06 08:12

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('alpha_business_app', '0010_alter_environmentconfig_marketplace'),
    ]

    operations = [
        migrations.AlterField(
            model_name='agentconfig',
            name='argument',
            field=models.CharField(default='', max_length=200),
        ),
        migrations.AlterField(
            model_name='environmentconfig',
            name='task',
            field=models.CharField(choices=[(1, 'training'), (2, 'agent_monitoring'), (3, 'exampleprinter')], max_length=14, null=True),
        ),
    ]
