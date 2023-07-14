<template>
    <div id="main_hmk">
      <div class="b1"><button type="button" id="b01" @click="create_data">手机号码</button></div>
      <div class="b1">
          <button  type="button" id="b02">身份证ID</button>
          <label>
            <input class="input_style" type="text" name="card_id" id="id_num" placeholder="请输入个数" v-model="num1">
          </label>
  
  
      </div>
      <div class="b1"><button type="button" id="b03"  @click="create_data">人名</button>
        <input class="input_style" type="text" name="name" id="id_num"  placeholder="请输入个数" v-model="num2">
      </div>
      <div class="b1"><button type="button" id="b07">清空输出</button></div>
  
      <div><textarea class="textera" id="result" v-model="info"></textarea></div>


    </div>
  
  </template>
  
  <script>
  
  import axios from 'axios';

  export default {name: "main_page",
  data() {
    return {
      num1: null, // 默认值设置为null
      num2: null,
      info: null,
    }
    },

    methods: {
        create_data(event) {
            if (event.target.id === "b01") {  //通过event.target.id，获取浏览器监听到的点击事件，并查看点击元素的id，通过比对id值判断触发哪个请求
            axios({
            url: "http://localhost:8000/firapp/phone"  //如果不指定method，默认发送get请求
            }).then(res => {
            this.info = res.data
             console.log(res)
            })
            }
            else if (event.target.id === "b03") {
            let payload = {
              num: this.num2
           }
            console.log(payload)
           axios({
            method: "get",
            params: payload,  //发送get请求，使用params关键字接收请求参数
            url: "http://localhost:8000/firapp/name"
            }).then(res => {
            this.info = res.data
             console.log(res)
           }).catch(err => {
            console.log(err)
           })
      }
        }
  }
}
  </script>
  
  <style scoped>
  .b1 {
    float: left;
  
    margin-right: 20px;
    margin-left: 50px;
    margin-top: 20px;
  }
  
  .textera {
    position:absolute;
    left: 60px;
    top: 80px;
    resize: both;  /*用户可调整元素的高度和宽度*/
    height: 244px;
    width: 823px;
  }
  
  .input_style {
    margin-left: 8px
  }
  </style>