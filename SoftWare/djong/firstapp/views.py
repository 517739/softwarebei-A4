from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, HttpResponseRedirect
from django.views.decorators.http import require_http_methods
import faker, json
from django.middleware.csrf import get_token

# Create your views here.
# 初始化，指定生成中文格式数据
def create_phone():
    fake = faker.Faker(locale='zh_CN')
    phones = [fake.phone_number() for _ in range(5)]
    return  " ".join(phones)


def phone(request):
    data = create_phone()
    return HttpResponse(data)

def create_id(num):
    fake = faker.Faker(locale='zh_CN')
    identity_ids = [fake.ssn() for i in range(int(num))]
    return " ".join(identity_ids)

@require_http_methods(['GET', 'POST'])
def id(request):
    num = request.POST.get("num")  # 如果"Content-type"="application/x-www-form-urlencoded"
    # num = json.loads(request.body).get("num") # 如果"Content-type"="application/json"
    print(num)
    if num == "" or num is None:
        data1 = create_id(5)
    else:
        data1 = create_id(num)
    return HttpResponse(data1)


def create_name(num):
    """生成姓名"""
    fake = faker.Faker(locale='zh_CN')
    names = [fake.name() for i in range(int(num))]
    return " ".join(names)


def name(request):
    num = request.GET.get("num")
    print(num)
    if num == "" or num is None:
        data = create_name(20)
    else:
        data = create_name(num)
    return HttpResponse(data)


def create_name(num):
    """生成姓名"""
    fake = faker.Faker(locale='zh_CN')
    names = [fake.name() for i in range(int(num))]  # 生成多个
    return " ".join(names)


def name(request):
    """
    生成姓名的视图方法
    :param request:
    :return:
    """
    num = request.GET.get("num")
    # print(num)
    if num == "" or num is None:
        data = create_name(20)
    else:
        data = create_name(num)
    return HttpResponse(data)


def main():
    return 0


if __name__ == '__main__':
    main()
