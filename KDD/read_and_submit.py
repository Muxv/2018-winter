# coding: utf-8

def read_data(data_type ,city, start_time, end_time):
    import requests
    """
    data_type :"airquality", "meteorology"
    city: "bj" , "ld", "bj_grid", "ld_grid"
    start/send_time :"YEAR-MONTH-DATE-HOUR"  : 2018-4-1-1
    """
    
    if data_type in ["aq","me"]:
        if data_type == "aq":
            data_type = "airquality"
        else :
            data_type = "meteorology"
            
    for i in [data_type, city, start_time, end_time]:
        i = str(i)
    
    url = 'https://biendata.com/competition/{0}/{1}/{2}/{3}/2k0d1d8'.format(data_type,city,start_time,end_time)
    response= requests.get(url)
    with open("{0}_{1}_{2}-{3}.csv".format(city, data_type, start_time, end_time),"w") as f:
        f.write(response.text)

		
def submit_data(file_dir, description, file_name):
    import requests
    
    User_id = "MuxvShen"
    token = "dadfd55b35345cf7375f3218098f65fd8015034f65aa6d689aa1734bc87ada5e"
    files = {"files": open(file_dir,"rb")}

    data = {
        "user_id": User_id,
        "team_token": token,
        "description": description,
        "filename": file_name,
    }
    url = 'https://biendata.com/competition/kdd_2018_submit/'
    response = requests.post(url, files=files, data=data)
    
    print(response.text)

# submit_data("sample_submission.csv", description="It's just a test",file_name="sample_submission")



