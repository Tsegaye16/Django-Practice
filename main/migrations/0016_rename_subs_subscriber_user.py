# Generated by Django 4.2 on 2023-05-10 13:28

from django.db import migrations


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0015_subscription_subscriber'),
    ]

    operations = [
        migrations.RenameField(
            model_name='subscriber',
            old_name='subs',
            new_name='user',
        ),
    ]