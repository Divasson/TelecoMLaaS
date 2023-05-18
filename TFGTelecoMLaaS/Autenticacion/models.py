from django.db import models
from django.contrib.auth.models import AbstractBaseUser,BaseUserManager

# Create your models here.

class UsuarioManager(BaseUserManager):
    def create_user(self, email, username, password=None):
        if not email:
            raise ValueError('Usuarios deben tener email')
        if not username:
            raise ValueError('Usuarios deben tener nombre de usuario')
        user = self.model(
                email = self.normalize_email(email),
                username=username,
            )
        user.set_password(password)
        user.save(using=self._db)
        return user
    def create_superuser(self,email,username,password):
        user = self.create_user(
                email = self.normalize_email(email),
                username=username,
                password = password,
            )
        user.is_admin= True
        user.is_staff = True
        user.is_superuser = True
        
        user.save(using=self._db)
        return user
        

class Usuario(AbstractBaseUser):
    #required
    email = models.EmailField(verbose_name="email",max_length=100,unique=True)
    username = models.CharField(max_length=50,unique=True)
    date_joined = models.DateField(auto_now_add=True,verbose_name="date_joined")
    last_login = models.DateField(auto_now=True,verbose_name="last_login")
    is_admin = models.BooleanField(default=False)
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)
    is_superuser = models.BooleanField(default=False)
    
    #not required
    listaProyectos = []
    
    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS= ['username']
    
    objects = UsuarioManager()
    
    def __str__(self):
        return str(self.email)+", "+ str(self.username)
    
    def has_perm(self,perm,obj=None):
        return self.is_admin
    
    def has_module_perms(self,app_label):
        return True
    
    
    
    
    #user = models.OneToOneField(User,on_delete=models.CASCADE,null=True)
    
    