# Generated by Django 4.2 on 2023-05-06 06:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0009_subplan_subplanfeature'),
    ]

    operations = [
        migrations.AddField(
            model_name='subplan',
            name='highlight_status',
            field=models.BooleanField(default=False, null=True),
        ),
    ]