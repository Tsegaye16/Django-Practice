# Generated by Django 4.2 on 2023-05-06 08:45

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0010_subplan_highlight_status'),
    ]

    operations = [
        migrations.AlterField(
            model_name='subplanfeature',
            name='subplan',
            field=models.ForeignKey(null=True, on_delete=django.db.models.deletion.CASCADE, to='main.subplan'),
        ),
    ]
