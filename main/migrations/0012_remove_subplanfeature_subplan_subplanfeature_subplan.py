# Generated by Django 4.2 on 2023-05-06 08:46

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0011_alter_subplanfeature_subplan'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='subplanfeature',
            name='subplan',
        ),
        migrations.AddField(
            model_name='subplanfeature',
            name='subplan',
            field=models.ManyToManyField(to='main.subplan'),
        ),
    ]
