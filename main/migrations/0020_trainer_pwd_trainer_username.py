# Generated by Django 4.2.1 on 2023-05-24 12:58

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0019_trainer_alter_subscriber_img_and_more'),
    ]

    operations = [
        migrations.AddField(
            model_name='trainer',
            name='pwd',
            field=models.CharField(max_length=30, null=True),
        ),
        migrations.AddField(
            model_name='trainer',
            name='username',
            field=models.CharField(max_length=100, null=True),
        ),
    ]