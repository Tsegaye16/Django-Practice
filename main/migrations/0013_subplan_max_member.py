# Generated by Django 4.2 on 2023-05-09 13:57

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0012_remove_subplanfeature_subplan_subplanfeature_subplan'),
    ]

    operations = [
        migrations.AddField(
            model_name='subplan',
            name='max_member',
            field=models.IntegerField(null=True),
        ),
    ]
