# Generated by Django 4.2 on 2023-05-04 23:56

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('main', '0004_page'),
    ]

    operations = [
        migrations.CreateModel(
            name='Faq',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('quest', models.TextField()),
                ('ans', models.TextField()),
            ],
        ),
    ]
