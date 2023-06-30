from django.db import models
from django.utils.html import mark_safe
from django.contrib.auth.models import User
from django.db.models.signals import post_save
from django.dispatch import receiver


#banners
class Banners(models.Model):
    img=models.ImageField(upload_to="banners/")
    alt_text=models.TextField(max_length=150)

    def __str__(self) -> str:
        return self.alt_text
    
    def image_tag(self):
        return mark_safe('<img src="%s" width="80"/>' % (self.img.url))
# Create your models here.

class Service(models.Model):
    title=models.CharField(max_length=150)
    detail=models.TextField()
    img=models.ImageField(upload_to="services/",null=True)

    def __str__(self) -> str:
        return self.title
    
    def image_tag(self):
        return mark_safe('<img src="%s" width="80"/>' % (self.img.url))

class Page(models.Model):
    title=models.CharField(max_length=200)
    detail=models.TextField()

    def __str__(self):
        return self.title

class Faq(models.Model):
    quest=models.TextField()
    ans=models.TextField()

    def __str__(self):
        return self.quest
    
class Enquiry(models.Model):
    full_name=models.CharField(max_length=150)
    email=models.CharField(max_length=150)
    detail=models.TextField()
    send_time=models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return self.full_name


class Gallery(models.Model):
    
    title=models.CharField(max_length=150)
    detail=models.TextField()
    img=models.ImageField(upload_to="gallery/",null=True)

    def __str__(self):
        return self.title
    
    def image_tag(self):
        return mark_safe('<img src="%s" width="80" /> '% (self.img.url))
    
class GalleryImage(models.Model):
    gallery=models.ForeignKey(Gallery,on_delete=models.CASCADE,null=True)
    alt_text=models.CharField(max_length=150)
    img=models.ImageField(upload_to="gallery_imgs/",null=True)

    def __str__(self):
        return self.alt_text
    
    def image_tag(self):
        return mark_safe('<img src="%s" width="80" />'%(self.img.url))
    
class SubPlan(models.Model):
    title=models.CharField(max_length=150)
    price=models.IntegerField()
    max_member=models.IntegerField(null=True)
    highlight_status=models.BooleanField(default=False,null=True)

    def __str__(self):
        return self.title
    
class SubPlanFeature(models.Model):
    #subplan=models.ForeignKey(SubPlan,on_delete=models.CASCADE,null=True)
    subplan=models.ManyToManyField(SubPlan)
    title=models.CharField(max_length=150)

    def __str__(self):
        return self.title
    

class PlanDiscount(models.Model):
    subplan=models.ForeignKey(SubPlan,on_delete=models.CASCADE, null=True)
    total_months=models.IntegerField()
    total_discount=models.IntegerField()

    def __str__(self):
        return str(self.total_months)
    

class Subscriber(models.Model):
    user=models.ForeignKey(User,on_delete=models.CASCADE,null=True)
    mobile=models.CharField(max_length=20)
    address=models.TextField()
    img=models.ImageField(upload_to="subs/",null=True)
    def __str__(self):
        return str(self.user)
    def image_tag(self):
        if self.img:        
            return mark_safe('<img src="%s" width="80" />' % (self.img.url))
        else:
            return "No-Image"
@receiver(post_save,sender=User)
def create_subscriber(sender,instance,created,**kwrags):
    if created:
        Subscriber.objects.create(user=instance)


class Subscription(models.Model):
    user=models.ForeignKey(User, on_delete=models.CASCADE,null=True)
    plan=models.ForeignKey(SubPlan, on_delete=models.CASCADE,null=True)
    price=models.CharField(max_length=50)



from imagekit.models import ProcessedImageField
from imagekit.processors import ResizeToFill

class Trainer(models.Model):
    full_name = models.CharField(max_length=100)
    username = models.CharField(max_length=100, null=True)
    pwd = models.CharField(max_length=30, null=True)
    mobile = models.CharField(max_length=20)
    address = models.TextField()
    is_active = models.BooleanField(default=False)
    detail = models.TextField()
    img = ProcessedImageField(upload_to="trainer/",
                              format='JPEG',
                              options={'quality': 90},
                              processors=[ResizeToFill(300, 300)])

    def __str__(self) -> str:
        return str(self.full_name)
    
    def image_tag(self):
        if self.img:
            return mark_safe('<img src="%s" width="80" />' % (self.img.url))
        else:
            return 'no-image'

# class Trainer(models.Model):
#     full_name=models.CharField(max_length=100)
#     username=models.CharField(max_length=100,null=True)
#     pwd=models.CharField(max_length=30,null=True)
#     mobile=models.CharField(max_length=20)
#     address=models.TextField()
#     is_active=models.BooleanField(default=False)
#     detail=models.TextField()
#     img = models.ImageField(upload_to="trainer/", blank=True, null=True)
   

    def __str__(self) -> str:
        return str(self.full_name)
    
    def image_tag(self):
        if self.img:
            return mark_safe('<img src="%s" width="80" />' % (self.img.url))
        else:
            return 'no-image'


class Notify(models.Model):
    notify_detail=models.TextField()
    read_by_user=models.ForeignKey(User,on_delete=models.CASCADE,null=True,blank=True)
    read_by_trainer=models.ForeignKey(Trainer,on_delete=models.CASCADE,null=True,blank=True)
    #status=models.BooleanField()

    def __str__(self) -> str:
        return str(self.notify_detail)
    

class NotifUserStatus(models.Model):
    notif=models.ForeignKey(Notify,on_delete=models.CASCADE)
    user=models.ForeignKey(User,on_delete=models.CASCADE)
    status=models.BooleanField(default=False)

class AssignSubscriber(models.Model):
    subscriber=models.ForeignKey(Subscriber,on_delete=models.CASCADE)
    trainer=models.ForeignKey(Trainer,on_delete=models.CASCADE)

    def __str__(self) -> str:
        return str(self.subscriber)