# Generated by Django 4.2 on 2023-05-04 08:33

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0002_banners'),
    ]

    operations = [
        migrations.AddField(
            model_name='service',
            name='img',
            field=models.ImageField(null=True, upload_to='services/'),
        ),
    ]
