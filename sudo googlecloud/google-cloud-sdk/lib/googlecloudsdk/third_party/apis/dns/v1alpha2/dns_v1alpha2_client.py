"""Generated client library for dns version v1alpha2."""
# NOTE: This file is autogenerated and should not be edited by hand.
from apitools.base.py import base_api
from googlecloudsdk.third_party.apis.dns.v1alpha2 import dns_v1alpha2_messages as messages


class DnsV1alpha2(base_api.BaseApiClient):
  """Generated client library for service dns version v1alpha2."""

  MESSAGES_MODULE = messages
  BASE_URL = u'https://www.googleapis.com/dns/v1alpha2/'

  _PACKAGE = u'dns'
  _SCOPES = [u'https://www.googleapis.com/auth/cloud-platform', u'https://www.googleapis.com/auth/cloud-platform.read-only', u'https://www.googleapis.com/auth/ndev.clouddns.readonly', u'https://www.googleapis.com/auth/ndev.clouddns.readwrite']
  _VERSION = u'v1alpha2'
  _CLIENT_ID = '1042881264118.apps.googleusercontent.com'
  _CLIENT_SECRET = 'x_Tw5K8nnjoRAqULM9PFAC2b'
  _USER_AGENT = 'x_Tw5K8nnjoRAqULM9PFAC2b'
  _CLIENT_CLASS_NAME = u'DnsV1alpha2'
  _URL_VERSION = u'v1alpha2'
  _API_KEY = None

  def __init__(self, url='', credentials=None,
               get_credentials=True, http=None, model=None,
               log_request=False, log_response=False,
               credentials_args=None, default_global_params=None,
               additional_http_headers=None):
    """Create a new dns handle."""
    url = url or self.BASE_URL
    super(DnsV1alpha2, self).__init__(
        url, credentials=credentials,
        get_credentials=get_credentials, http=http, model=model,
        log_request=log_request, log_response=log_response,
        credentials_args=credentials_args,
        default_global_params=default_global_params,
        additional_http_headers=additional_http_headers)
    self.changes = self.ChangesService(self)
    self.dnsKeys = self.DnsKeysService(self)
    self.managedZoneOperations = self.ManagedZoneOperationsService(self)
    self.managedZones = self.ManagedZonesService(self)
    self.projects = self.ProjectsService(self)
    self.resourceRecordSets = self.ResourceRecordSetsService(self)

  class ChangesService(base_api.BaseApiService):
    """Service class for the changes resource."""

    _NAME = u'changes'

    def __init__(self, client):
      super(DnsV1alpha2.ChangesService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      """Atomically update the ResourceRecordSet collection.

      Args:
        request: (DnsChangesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Change) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'POST',
        method_id=u'dns.changes.create',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/changes',
        request_field=u'change',
        request_type_name=u'DnsChangesCreateRequest',
        response_type_name=u'Change',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      """Fetch the representation of an existing Change.

      Args:
        request: (DnsChangesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Change) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.changes.get',
        ordered_params=[u'project', u'managedZone', u'changeId'],
        path_params=[u'changeId', u'managedZone', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/changes/{changeId}',
        request_field='',
        request_type_name=u'DnsChangesGetRequest',
        response_type_name=u'Change',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Enumerate Changes to a ResourceRecordSet collection.

      Args:
        request: (DnsChangesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ChangesListResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.changes.list',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'maxResults', u'pageToken', u'sortBy', u'sortOrder'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/changes',
        request_field='',
        request_type_name=u'DnsChangesListRequest',
        response_type_name=u'ChangesListResponse',
        supports_download=False,
    )

  class DnsKeysService(base_api.BaseApiService):
    """Service class for the dnsKeys resource."""

    _NAME = u'dnsKeys'

    def __init__(self, client):
      super(DnsV1alpha2.DnsKeysService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      """Fetch the representation of an existing DnsKey.

      Args:
        request: (DnsDnsKeysGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsKey) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.dnsKeys.get',
        ordered_params=[u'project', u'managedZone', u'dnsKeyId'],
        path_params=[u'dnsKeyId', u'managedZone', u'project'],
        query_params=[u'clientOperationId', u'digestType'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/dnsKeys/{dnsKeyId}',
        request_field='',
        request_type_name=u'DnsDnsKeysGetRequest',
        response_type_name=u'DnsKey',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Enumerate DnsKeys to a ResourceRecordSet collection.

      Args:
        request: (DnsDnsKeysListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (DnsKeysListResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.dnsKeys.list',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'digestType', u'maxResults', u'pageToken'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/dnsKeys',
        request_field='',
        request_type_name=u'DnsDnsKeysListRequest',
        response_type_name=u'DnsKeysListResponse',
        supports_download=False,
    )

  class ManagedZoneOperationsService(base_api.BaseApiService):
    """Service class for the managedZoneOperations resource."""

    _NAME = u'managedZoneOperations'

    def __init__(self, client):
      super(DnsV1alpha2.ManagedZoneOperationsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      """Fetch the representation of an existing Operation.

      Args:
        request: (DnsManagedZoneOperationsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.managedZoneOperations.get',
        ordered_params=[u'project', u'managedZone', u'operation'],
        path_params=[u'managedZone', u'operation', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/operations/{operation}',
        request_field='',
        request_type_name=u'DnsManagedZoneOperationsGetRequest',
        response_type_name=u'Operation',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Enumerate Operations for the given ManagedZone.

      Args:
        request: (DnsManagedZoneOperationsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZoneOperationsListResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.managedZoneOperations.list',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'maxResults', u'pageToken', u'sortBy'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/operations',
        request_field='',
        request_type_name=u'DnsManagedZoneOperationsListRequest',
        response_type_name=u'ManagedZoneOperationsListResponse',
        supports_download=False,
    )

  class ManagedZonesService(base_api.BaseApiService):
    """Service class for the managedZones resource."""

    _NAME = u'managedZones'

    def __init__(self, client):
      super(DnsV1alpha2.ManagedZonesService, self).__init__(client)
      self._upload_configs = {
          }

    def Create(self, request, global_params=None):
      """Create a new ManagedZone.

      Args:
        request: (DnsManagedZonesCreateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
      config = self.GetMethodConfig('Create')
      return self._RunMethod(
          config, request, global_params=global_params)

    Create.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'POST',
        method_id=u'dns.managedZones.create',
        ordered_params=[u'project'],
        path_params=[u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones',
        request_field=u'managedZone',
        request_type_name=u'DnsManagedZonesCreateRequest',
        response_type_name=u'ManagedZone',
        supports_download=False,
    )

    def Delete(self, request, global_params=None):
      """Delete a previously created ManagedZone.

      Args:
        request: (DnsManagedZonesDeleteRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZonesDeleteResponse) The response message.
      """
      config = self.GetMethodConfig('Delete')
      return self._RunMethod(
          config, request, global_params=global_params)

    Delete.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'DELETE',
        method_id=u'dns.managedZones.delete',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}',
        request_field='',
        request_type_name=u'DnsManagedZonesDeleteRequest',
        response_type_name=u'ManagedZonesDeleteResponse',
        supports_download=False,
    )

    def Get(self, request, global_params=None):
      """Fetch the representation of an existing ManagedZone.

      Args:
        request: (DnsManagedZonesGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZone) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.managedZones.get',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}',
        request_field='',
        request_type_name=u'DnsManagedZonesGetRequest',
        response_type_name=u'ManagedZone',
        supports_download=False,
    )

    def List(self, request, global_params=None):
      """Enumerate ManagedZones that have been created but not yet deleted.

      Args:
        request: (DnsManagedZonesListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ManagedZonesListResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.managedZones.list',
        ordered_params=[u'project'],
        path_params=[u'project'],
        query_params=[u'dnsName', u'maxResults', u'pageToken'],
        relative_path=u'projects/{project}/managedZones',
        request_field='',
        request_type_name=u'DnsManagedZonesListRequest',
        response_type_name=u'ManagedZonesListResponse',
        supports_download=False,
    )

    def Patch(self, request, global_params=None):
      """Update an existing ManagedZone. This method supports patch semantics.

      Args:
        request: (DnsManagedZonesPatchRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Patch')
      return self._RunMethod(
          config, request, global_params=global_params)

    Patch.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'PATCH',
        method_id=u'dns.managedZones.patch',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}',
        request_field=u'managedZoneResource',
        request_type_name=u'DnsManagedZonesPatchRequest',
        response_type_name=u'Operation',
        supports_download=False,
    )

    def Update(self, request, global_params=None):
      """Update an existing ManagedZone.

      Args:
        request: (DnsManagedZonesUpdateRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Operation) The response message.
      """
      config = self.GetMethodConfig('Update')
      return self._RunMethod(
          config, request, global_params=global_params)

    Update.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'PUT',
        method_id=u'dns.managedZones.update',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}/managedZones/{managedZone}',
        request_field=u'managedZoneResource',
        request_type_name=u'DnsManagedZonesUpdateRequest',
        response_type_name=u'Operation',
        supports_download=False,
    )

  class ProjectsService(base_api.BaseApiService):
    """Service class for the projects resource."""

    _NAME = u'projects'

    def __init__(self, client):
      super(DnsV1alpha2.ProjectsService, self).__init__(client)
      self._upload_configs = {
          }

    def Get(self, request, global_params=None):
      """Fetch the representation of an existing Project.

      Args:
        request: (DnsProjectsGetRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (Project) The response message.
      """
      config = self.GetMethodConfig('Get')
      return self._RunMethod(
          config, request, global_params=global_params)

    Get.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.projects.get',
        ordered_params=[u'project'],
        path_params=[u'project'],
        query_params=[u'clientOperationId'],
        relative_path=u'projects/{project}',
        request_field='',
        request_type_name=u'DnsProjectsGetRequest',
        response_type_name=u'Project',
        supports_download=False,
    )

  class ResourceRecordSetsService(base_api.BaseApiService):
    """Service class for the resourceRecordSets resource."""

    _NAME = u'resourceRecordSets'

    def __init__(self, client):
      super(DnsV1alpha2.ResourceRecordSetsService, self).__init__(client)
      self._upload_configs = {
          }

    def List(self, request, global_params=None):
      """Enumerate ResourceRecordSets that have been created but not yet deleted.

      Args:
        request: (DnsResourceRecordSetsListRequest) input message
        global_params: (StandardQueryParameters, default: None) global arguments
      Returns:
        (ResourceRecordSetsListResponse) The response message.
      """
      config = self.GetMethodConfig('List')
      return self._RunMethod(
          config, request, global_params=global_params)

    List.method_config = lambda: base_api.ApiMethodInfo(
        http_method=u'GET',
        method_id=u'dns.resourceRecordSets.list',
        ordered_params=[u'project', u'managedZone'],
        path_params=[u'managedZone', u'project'],
        query_params=[u'maxResults', u'name', u'pageToken', u'type'],
        relative_path=u'projects/{project}/managedZones/{managedZone}/rrsets',
        request_field='',
        request_type_name=u'DnsResourceRecordSetsListRequest',
        response_type_name=u'ResourceRecordSetsListResponse',
        supports_download=False,
    )
