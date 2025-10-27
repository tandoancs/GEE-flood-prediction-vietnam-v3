import ee
import pandas as pd
import time
import os

# ee.Authenticate() # Bỏ comment và chạy lần đầu tiên
# ee.Initialize(project='your-ee-project-id') # Thay 'your-ee-project-id' bằng ID dự án GEE của bạn
ee.Initialize()


# --- CONFIGURATION ---
FLOOD_DATA_TRAIN = os.path.join('..', 'data', 'flood_data_train.csv')
FLOOD_DATA_VALIDATION = os.path.join('..', 'data', 'flood_data_validation.csv')
FLOOD_DATA_TEST = os.path.join('..', 'data', 'flood_data_test.csv')

# Ưu tiên 1A: Phạm vi Toàn quốc sử dụng FAO GAUL
ADMIN_BOUNDARIES_ID = 'FAO/GAUL/2015/level1'
COUNTRY_NAME = 'Viet Nam'

# Ưu tiên 1B: Nhiều Sự kiện Lũ lụt Lịch sử
# Tên tỉnh phải khớp với thuộc tính 'ADM1_NAME' trong bộ dữ liệu GAUL.

FLOOD_EVENTS = [
    # Tập xác thực (validation set)
    {'id': 'FL_VAL_2022', 'start': '2022-10-01', 'end': '2022-10-05', 
     'provinces': ['Nghe An', 'Ha Tinh'], 'purpose': 'validation'},
    
    # Tập huấn luyện (training set)
    {'id': 'FL2021_10', 'start': '2021-10-21', 'end': '2021-10-28', 
     'provinces': ['Quang Binh', 'Quang Tri'], 'purpose': 'training'},
    {'id': 'FL2020_10', 'start': '2020-10-06', 'end': '2020-11-17', 
     'provinces': ['Thua Thien - Hue', 'Da Nang City', 'Quang Nam'], 'purpose': 'training'},
    {'id': 'FL2015_07', 'start': '2015-07-26', 'end': '2015-07-28', 
     'provinces': ['Quang Ninh'], 'purpose': 'training'},
    
    # Tập kiểm tra cuối cùng (hold-out test set)
    # SỬA LỖI KHÔNG NHẤT QUÁN: Đã đổi 'Da Nang' thành 'Da Nang City' để khớp
    {'id': 'FL2023_10', 'start': '2023-10-13', 'end': '2023-10-17', 
     'provinces': ['Da Nang City', 'Quang Nam'], 'purpose': 'testing'}
]


# GEE Asset IDs
# Ảnh radar (SAR) từ vệ tinh Sentinel-1. 
# Dữ liệu này rất quan trọng vì nó có thể "nhìn" xuyên qua mây và được dùng để phát hiện nước lũ.
SENTINEL1_ID = 'COPERNICUS/S1_GRD'

# Dữ liệu độ cao (Digital Elevation Model
DEM_ID = 'CGIAR/SRTM90_V4'

# Dữ liệu về sông ngòi.
RIVER_ID = 'WWF/HydroSHEDS/v1/FreeFlowingRivers'

# Dữ liệu về tỷ lệ sét trong đất.
SOIL_CLAY_ID = 'OpenLandMap/SOL/SOL_CLAY-WFRACTION_USDA-3A1A1A_M/v02'

# Tham số Lấy mẫu

# Số lượng điểm dữ liệu cần lấy cho mỗi tỉnh trong mỗi sự kiện (1000 điểm).
SAMPLES_PER_PROVINCE_PER_EVENT = 1000 

# Độ phân giải không gian (30 mét). Mọi dữ liệu sẽ được xử lý ở kích thước pixel 30x30m.
SCALE = 30  

# Cửa sổ thời gian để lấy dữ liệu mưa (14 ngày).
TIME_WINDOW_DAYS = 14 

# Ưu tiên 1C: Tham số Lấy mẫu Thông minh

# Thay vì lấy mẫu ngẫu nhiên bao gồm núi
# Chỉ tập trung vào các khu vực có nguy cơ ngập lụt cao: 
# những nơi có độ cao dưới 50m VÀ gần sông (trong vòng 2000m).
ELEVATION_THRESHOLD = 50 
RIVER_BUFFER = 2000 

# --- HELPER FUNCTIONS ---
def get_vietnam_provinces():
    """Lấy danh sách tất cả tên tỉnh và hình học của chúng ở Việt Nam."""
    admin_boundaries = ee.FeatureCollection(ADMIN_BOUNDARIES_ID)
    vietnam_boundaries = admin_boundaries.filter(ee.Filter.eq('ADM0_NAME', COUNTRY_NAME))
    
    province_info = vietnam_boundaries.getInfo()['features']
    
    provinces = {}
    for feature in province_info:
        props = feature['properties']
        geom = ee.Geometry(feature['geometry'])
        province_name = props.get('ADM1_NAME') 
        if province_name:
            provinces[province_name] = geom
            
    print(f"Đã tải {len(provinces)} tỉnh cho {COUNTRY_NAME}.")
    
    # In ra tất cả các tên tỉnh tìm thấy để bạn có thể sao chép chính xác
    print(f"Tên tỉnh có sẵn: {list(provinces.keys())}")
    
    return provinces

def create_risk_mask(region):
    """Tạo một mặt nạ nhị phân cho các khu vực dễ bị ngập lụt."""
    dem = ee.Image(DEM_ID).select('elevation')
    rivers = ee.FeatureCollection(RIVER_ID).filterBounds(region)
    
    elevation_mask = dem.lte(ELEVATION_THRESHOLD)
    
    river_distance = ee.Image(0).byte().paint(rivers, 1).fastDistanceTransform(
        RIVER_BUFFER, 'meters'
    ).select('distance')
    
    river_mask = river_distance.lte(RIVER_BUFFER)
    
    risk_mask = elevation_mask.And(river_mask)
    return risk_mask.selfMask() 

def get_flood_data(region, start_date, end_date):
    """Xác định các khu vực bị ngập bằng dữ liệu Sentinel-1 SAR."""
    s1 = ee.ImageCollection(SENTINEL1_ID) \
      .filterBounds(region) \
      .filter(ee.Filter.listContains('transmitterReceiverPolarisation', 'VV')) \
      .filter(ee.Filter.eq('instrumentMode', 'IW'))

    before_start = ee.Date(start_date).advance(-3, 'month')
    before_end = ee.Date(start_date).advance(-1, 'month')

    s1_before_coll = s1.filterDate(before_start, before_end)
    s1_during_coll = s1.filterDate(start_date, end_date)

    default_image = ee.Image(0).toFloat().rename('VV').clip(region)

    before_flood = ee.Image(ee.Algorithms.If(
        s1_before_coll.size().gt(0),
        s1_before_coll.select('VV').mosaic().clip(region),
        default_image
    ))
    
    during_flood = ee.Image(ee.Algorithms.If(
        s1_during_coll.size().gt(0),
        s1_during_coll.select('VV').mosaic().clip(region), 
        default_image
    ))

    ratio = before_flood.divide(during_flood.add(1e-6))
    
    flooded = ratio.gt(1.25)
    return flooded.rename('flooded')


def get_static_features(region):
    """Trích xuất các đặc trưng địa hình và đất tĩnh."""
    dem = ee.Image(DEM_ID).select('elevation').clip(region)
    slope = ee.Terrain.slope(dem).rename('slope')
    aspect = ee.Terrain.aspect(dem).rename('aspect')
    clay = ee.Image(SOIL_CLAY_ID).select('b0').clip(region).rename('clay_content')
    
    return ee.Image.cat([dem, slope, aspect, clay])

def get_dynamic_features(region, date):
    """Trích xuất dữ liệu mưa chuỗi thời gian (GPM) cho một ngày nhất định."""
    start_date = date.advance(-TIME_WINDOW_DAYS, 'day')
    gpm = ee.ImageCollection('NASA/GPM_L3/IMERG_V07') \
      .filterBounds(region) \
      .filterDate(start_date, date) \
      .select('precipitation')
    
    # Tạo một danh sách các ảnh, mỗi ảnh cho một ngày trong cửa sổ thời gian
    def create_daily_image(day_offset):
        day_offset_num = ee.Number(day_offset)
        target_date = date.advance(day_offset_num.multiply(-1), 'day')
        
        # SỬA LỖI QUAN TRỌNG: Xử lý các ngày không có dữ liệu
        
        # Lọc collection cho ngày đó
        daily_coll = gpm.filterDate(target_date, target_date.advance(1, 'day'))
        
        # Tính tổng, sẽ là 0-band nếu collection rỗng
        daily_precip_sum = daily_coll.sum()
        
        # Tạo ảnh 0-value mặc định với đúng tên band (trước khi rename)
        default_precip = ee.Image(0).toFloat().rename('precipitation')
        
        # Sử dụng ee.Algorithms.If để chọn
        # Nếu collection có size > 0, dùng sum. Nếu không, dùng ảnh 0-value.
        daily_precip = ee.Image(ee.Algorithms.If(
            daily_coll.size().gt(0),
            daily_precip_sum,
            default_precip
        ))
        
        # Đảm bảo tên band hợp lệ
        band_name = ee.String('precip_day_').cat(day_offset_num.format('%02d'))
        
        # Đổi tên band 'precipitation' (từ sum hoặc default) thành tên band cuối cùng
        return daily_precip.rename(band_name) 

    day_indices = ee.List.sequence(0, TIME_WINDOW_DAYS - 1)

    # --- SỬA LỖI TÊN CỘT (BUG FIX) ---
    # `ee.Image.cat()` không thể xử lý trực tiếp một ee.List (server-side list).
    # Chúng ta phải lặp (iterate) qua danh sách và tự addBands.
    
    # .map() trên ee.List trả về một ee.List.
    daily_images_list = day_indices.map(create_daily_image)
    
    # Lấy ảnh đầu tiên làm cơ sở (phải ép kiểu về ee.Image)
    first_image = ee.Image(daily_images_list.get(0))
    
    # Lấy phần còn lại của danh sách (từ chỉ số 1 trở đi)
    remaining_images = daily_images_list.slice(1)
    
    # Định nghĩa một hàm (phía máy chủ) để_gộp các ảnh
    def add_band_to_image(image, previous_image):
        # 'previous_image' là kết quả tích lũy từ lần lặp trước (phải ép kiểu)
        # 'image' là mục hiện tại từ 'remaining_images' (phải ép kiểu)
        return ee.Image(previous_image).addBands(ee.Image(image))

    # Lặp qua phần còn lại của danh sách và gộp chúng vào ảnh đầu tiên
    # 'first_image' là giá trị khởi tạo
    combined_dynamic_image = ee.Image(remaining_images.iterate(add_band_to_image, first_image))
    
    # Giờ đây, hàm này được đảm bảo trả về một ảnh có 14 band với tên chính xác
    return combined_dynamic_image

# --- MAIN WORKFLOW ---

def main():
    all_provinces = get_vietnam_provinces()
    
    all_data = []

    for event in FLOOD_EVENTS:
        event_id = event['id']
        start_date = event['start']
        end_date = event['end']
        purpose = event['purpose']
        
        print(f"\n--- Đang xử lý Sự kiện: {event_id} ({start_date} đến {end_date}) ---")
        
        for province_name in event['provinces']:
            if province_name not in all_provinces:
                print(f"Cảnh báo: Tỉnh '{province_name}' không tìm thấy trong bộ dữ liệu GAUL. Bỏ qua.")
                continue
            
            print(f"  Đang xử lý Tỉnh: {province_name}")
            
            try:
                study_region = all_provinces[province_name]
                risk_mask = create_risk_mask(study_region)
                flooded_map = get_flood_data(study_region, start_date, end_date)
                static_features = get_static_features(study_region)
                
                event_mid_date = ee.Date(start_date).advance(
                    ee.Date(end_date).difference(ee.Date(start_date), 'day').divide(2), 'day'
                )
                dynamic_features = get_dynamic_features(study_region, event_mid_date)

                combined_image = ee.Image.cat([
                    flooded_map.unmask(0), 
                    static_features, 
                    dynamic_features
                ])

                samples = combined_image.updateMask(risk_mask).stratifiedSample(
                    numPoints=SAMPLES_PER_PROVINCE_PER_EVENT,
                    classBand='flooded',
                    region=study_region,
                    scale=SCALE,
                    geometries=True,
                    tileScale=4 
                )

                feature_list = samples.getInfo()['features']
                
                for feature in feature_list:
                    props = feature['properties']
                    coords = feature['geometry']['coordinates']
                    
                    row = {
                        'event_id': event_id,
                        'province': province_name,
                        'purpose': purpose,
                        'lon': coords[0], 
                        'lat': coords[1], 
                        **props
                    }
                    all_data.append(row)
                
                print(f"    -> Lấy mẫu thành công {len(feature_list)} điểm.")
            
            except Exception as e:
                print(f"    -> LỖI khi xử lý {province_name}: {e}")
                time.sleep(10)

    if not all_data:
        print("\nKhông có dữ liệu nào được thu thập. Kết thúc.")
        return

    df = pd.DataFrame(all_data)
    
    df_train = df[df['purpose'] == 'training']
    df_val = df[df['purpose'] == 'validation']
    df_test = df[df['purpose'] == 'testing']

    df_train.to_csv(FLOOD_DATA_TRAIN, index=False)
    df_val.to_csv(FLOOD_DATA_VALIDATION, index=False)
    df_test.to_csv(FLOOD_DATA_TEST, index=False)

    print(f"\n--- Chuẩn bị Dữ liệu Hoàn tất ---")
    print(f"Dữ liệu huấn luyện đã được lưu vào flood_data_train.csv ({len(df_train)} hàng)")
    print(f"Dữ liệu xác thực đã được lưu vào flood_data_validation.csv ({len(df_val)} hàng)")
    print(f"Dữ liệu kiểm tra đã được lưu vào flood_data_test.csv ({len(df_test)} hàng)")

if __name__ == '__main__':
    main()

